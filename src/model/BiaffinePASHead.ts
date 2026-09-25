/**
 * Japanese 10-Case Dynamic Biaffine PAS (Predicate-Argument Structure) Projection Head
 * Implements bilinear inner-product scoring, Wasm SIMD / ONNX Runtime tensor post-processing,
 * score normalization for zero pronouns and subject-predicate distortion candidates.
 */

export const JAPANESE_PAS_CASES = [
  'ガ',
  'ガ２',
  'ヲ',
  'ニ',
  'ト',
  'デ',
  'カラ',
  'ヨリ',
  'ヘ',
  'マデ'
] as const;

export type PASCase = typeof JAPANESE_PAS_CASES[number];

export interface BiaffinePASConfig {
  hiddenDim: number; // e.g., 256
  numCases: number;  // 10
  cases: readonly string[];
}

export interface BiaffinePASWeights {
  wPred: Float32Array; // [hiddenDim x hiddenDim]
  wArg: Float32Array;  // [hiddenDim x hiddenDim]
  uCases: Float32Array; // [numCases x hiddenDim x hiddenDim]
}

export interface PASArgument {
  caseName: string;
  argIndex: number | null; // null indicates zero pronoun (omitted argument)
  argText?: string;
  score: number;
  isZeroPronoun: boolean;
}

export interface PASPredicateRelation {
  predicateIndex: number;
  predicateText?: string;
  arguments: PASArgument[];
}

export interface NormalizationOptions {
  temperature?: number;          // Default: 1.0
  threshold?: number;            // Confidence threshold, default 0.15
  zeroPronounThreshold?: number; // Threshold below which subject is considered zero pronoun, default 0.3
  applySoftmax?: boolean;        // Whether to apply softmax along argument dimension, default true
  distortionPenalty?: number;    // Penalty factor for subject-predicate distortion candidates, default 0.1
}

export interface ExtractionOptions extends NormalizationOptions {
  sequenceText?: string;
  predicateIndices?: number[];   // If specified, extract only for these predicate positions
  maxSpanLength?: number;        // Maximum character length for argument span aggregation
}

export class BiaffinePASHead {
  public readonly config: BiaffinePASConfig;
  private weights: BiaffinePASWeights;

  constructor(
    config?: Partial<BiaffinePASConfig>,
    weights?: Partial<BiaffinePASWeights>
  ) {
    this.config = {
      hiddenDim: config?.hiddenDim ?? 256,
      numCases: config?.numCases ?? JAPANESE_PAS_CASES.length,
      cases: config?.cases ?? JAPANESE_PAS_CASES,
    };

    const d = this.config.hiddenDim;
    const c = this.config.numCases;

    this.weights = {
      wPred: weights?.wPred ?? new Float32Array(d * d),
      wArg: weights?.wArg ?? new Float32Array(d * d),
      uCases: weights?.uCases ?? new Float32Array(c * d * d),
    };

    if (!weights) {
      this.initDefaultWeights();
    }
  }

  /**
   * Initialize weights using Xavier uniform distribution
   */
  public initDefaultWeights(): void {
    const d = this.config.hiddenDim;
    const c = this.config.numCases;

    const limitW = Math.sqrt(6 / (d + d));
    for (let i = 0; i < d * d; i++) {
      this.weights.wPred[i] = (Math.random() * 2 - 1) * limitW;
      this.weights.wArg[i] = (Math.random() * 2 - 1) * limitW;
    }

    const limitU = Math.sqrt(6 / (d + d));
    for (let i = 0; i < c * d * d; i++) {
      this.weights.uCases[i] = (Math.random() * 2 - 1) * limitU;
    }
  }

  /**
   * Set explicitly trained or exported weights
   */
  public setWeights(weights: Partial<BiaffinePASWeights>): void {
    if (weights.wPred) this.weights.wPred = weights.wPred;
    if (weights.wArg) this.weights.wArg = weights.wArg;
    if (weights.uCases) this.weights.uCases = weights.uCases;
  }

  /**
   * Get current weight parameters
   */
  public getWeights(): BiaffinePASWeights {
    return this.weights;
  }

  /**
   * Compute dynamic bilinear inner-product PAS score matrix
   * Input H: [L x hiddenDim] or flat Float32Array of length (L * hiddenDim)
   * Output: 3D score tensor [C, L, L] where C=10 (cases), L=seqLen
   */
  public forward(
    hiddenStates: number[][] | Float32Array,
    seqLen?: number
  ): number[][][] {
    const d = this.config.hiddenDim;
    const c = this.config.numCases;

    let L: number;
    let hMatrix: Float32Array;

    if (hiddenStates instanceof Float32Array) {
      L = seqLen ?? Math.floor(hiddenStates.length / d);
      hMatrix = hiddenStates;
    } else {
      L = hiddenStates.length;
      hMatrix = new Float32Array(L * d);
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < d; j++) {
          hMatrix[i * d + j] = hiddenStates[i][j] ?? 0;
        }
      }
    }

    // 1. Linear projections: h_pred = H * W_pred, h_arg = H * W_arg
    const hPred = new Float32Array(L * d);
    const hArg = new Float32Array(L * d);

    for (let i = 0; i < L; i++) {
      for (let col = 0; col < d; col++) {
        let sumPred = 0;
        let sumArg = 0;
        for (let k = 0; k < d; k++) {
          const hVal = hMatrix[i * d + k];
          sumPred += hVal * this.weights.wPred[k * d + col];
          sumArg += hVal * this.weights.wArg[k * d + col];
        }
        hPred[i * d + col] = sumPred;
        hArg[i * d + col] = sumArg;
      }
    }

    // 2. Bilinear Tensor Inner-Product: S[case][i][j] = (h_pred[i])^T * U_case * h_arg[j]
    const scores: number[][][] = Array.from({ length: c }, () =>
      Array.from({ length: L }, () => new Array(L).fill(0))
    );

    // Intermediate projection for each case & predicate i: temp[col] = (hPred[i])^T * U_case
    const temp = new Float32Array(d);

    for (let caseIdx = 0; caseIdx < c; caseIdx++) {
      const uOffset = caseIdx * d * d;

      for (let i = 0; i < L; i++) {
        // temp = hPred[i] * U_case
        for (let col = 0; col < d; col++) {
          let sum = 0;
          for (let row = 0; row < d; row++) {
            sum += hPred[i * d + row] * this.weights.uCases[uOffset + row * d + col];
          }
          temp[col] = sum;
        }

        // score = temp . hArg[j]
        for (let j = 0; j < L; j++) {
          let score = 0;
          for (let col = 0; col < d; col++) {
            score += temp[col] * hArg[j * d + col];
          }
          scores[caseIdx][i][j] = score;
        }
      }
    }

    return scores;
  }

  /**
   * Parse flat Float32Array output buffer from Wasm SIMD or ONNX Runtime execution
   * Shape: [numCases x seqLen x seqLen]
   */
  public parseRawTensor(
    flatBuffer: Float32Array | number[],
    seqLen: number,
    numCases?: number
  ): number[][][] {
    const C = numCases ?? this.config.numCases;
    const L = seqLen;

    const scores: number[][][] = Array.from({ length: C }, () =>
      Array.from({ length: L }, () => new Array(L).fill(0))
    );

    for (let caseIdx = 0; caseIdx < C; caseIdx++) {
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < L; j++) {
          const flatIdx = caseIdx * L * L + i * L + j;
          scores[caseIdx][i][j] = flatBuffer[flatIdx] ?? 0;
        }
      }
    }

    return scores;
  }

  /**
   * Score normalization for zero pronouns (主語省略) and subject-predicate distortion candidates.
   * Normalizes scores along argument dimension j using temperature-scaled softmax/sigmoid,
   * handles score sinking for unknown words, and computes zero-pronoun implicit argument probability.
   */
  public normalizeScores(
    rawScores: number[][][] | Float32Array,
    seqLen?: number,
    options?: NormalizationOptions
  ): {
    normalizedScores: number[][][];
    zeroPronounScores: number[][]; // [numCases][predicate_i] -> score for zero pronoun
  } {
    const temp = options?.temperature ?? 1.0;
    const applySoftmax = options?.applySoftmax ?? true;
    const distortionPenalty = options?.distortionPenalty ?? 0.1;

    let scores: number[][][];
    if (rawScores instanceof Float32Array) {
      const L = seqLen ?? Math.floor(Math.sqrt(rawScores.length / this.config.numCases));
      scores = this.parseRawTensor(rawScores, L);
    } else {
      scores = rawScores;
    }

    const C = scores.length;
    const L = scores[0]?.length ?? 0;

    const normalizedScores: number[][][] = Array.from({ length: C }, () =>
      Array.from({ length: L }, () => new Array(L).fill(0))
    );
    const zeroPronounScores: number[][] = Array.from({ length: C }, () => new Array(L).fill(0));

    for (let c = 0; c < C; c++) {
      for (let i = 0; i < L; i++) {
        const rawRow = scores[c][i];

        // Apply temperature scaling
        const scaledRow = rawRow.map((s) => s / Math.max(temp, 1e-5));

        // Subject-predicate distortion detection (skewness/entropy check)
        // If distance |i - j| is large or subject-predicate candidate scores are distorted, apply penalty
        const isSubjectCase = this.config.cases[c] === 'ガ' || this.config.cases[c] === 'ガ２';
        const adjustedRow = scaledRow.map((s, j) => {
          let score = s;
          const dist = Math.abs(i - j);
          if (isSubjectCase && dist > 32) {
            score -= distortionPenalty * (dist - 32);
          }
          return score;
        });

        if (applySoftmax) {
          const maxVal = Math.max(...adjustedRow);
          const exps = adjustedRow.map((s) => Math.exp(s - maxVal));
          const sumExp = exps.reduce((a, b) => a + b, 0);

          for (let j = 0; j < L; j++) {
            normalizedScores[c][i][j] = exps[j] / Math.max(sumExp, 1e-9);
          }

          // Zero pronoun score calculation: probability that none of explicit tokens fill this case
          const maxExplicitScore = Math.max(...normalizedScores[c][i]);
          zeroPronounScores[c][i] = Math.max(0, 1.0 - maxExplicitScore);
        } else {
          // Sigmoid normalization
          for (let j = 0; j < L; j++) {
            const sig = 1 / (1 + Math.exp(-adjustedRow[j]));
            normalizedScores[c][i][j] = sig;
          }
          const maxExplicitScore = Math.max(...normalizedScores[c][i]);
          zeroPronounScores[c][i] = Math.max(0, 1.0 - maxExplicitScore);
        }
      }
    }

    return { normalizedScores, zeroPronounScores };
  }

  /**
   * Utility for extracting predicate-argument structure relations and text spans
   */
  public extractRelations(
    inputScores: number[][][] | Float32Array,
    options?: ExtractionOptions
  ): PASPredicateRelation[] {
    const seqText = options?.sequenceText;
    const threshold = options?.threshold ?? 0.15;
    const zeroTh = options?.zeroPronounThreshold ?? 0.3;
    const targetPreds = options?.predicateIndices;

    const { normalizedScores, zeroPronounScores } = this.normalizeScores(
      inputScores,
      seqText ? seqText.length : undefined,
      options
    );

    const C = normalizedScores.length;
    const L = normalizedScores[0]?.length ?? 0;

    const results: PASPredicateRelation[] = [];
    const predIndices = targetPreds ?? Array.from({ length: L }, (_, i) => i);

    for (const i of predIndices) {
      if (i < 0 || i >= L) continue;

      const predText = seqText ? seqText.slice(i, Math.min(i + 4, L)) : undefined;
      const args: PASArgument[] = [];

      for (let c = 0; c < C; c++) {
        const caseName = this.config.cases[c] ?? `case_${c}`;
        const row = normalizedScores[c][i];

        let bestArgIdx = -1;
        let bestScore = -1;

        for (let j = 0; j < L; j++) {
          if (row[j] > bestScore) {
            bestScore = row[j];
            bestArgIdx = j;
          }
        }

        const isSubjectCase = caseName === 'ガ' || caseName === 'ガ２';
        const zpScore = zeroPronounScores[c][i];

        // Check if argument is zero pronoun (omitted subject)
        if (isSubjectCase && (bestScore < threshold || zpScore >= zeroTh)) {
          args.push({
            caseName,
            argIndex: null,
            score: zpScore,
            isZeroPronoun: true,
          });
        } else if (bestScore >= threshold && bestArgIdx >= 0) {
          const argText = seqText ? seqText[bestArgIdx] : undefined;
          args.push({
            caseName,
            argIndex: bestArgIdx,
            argText,
            score: bestScore,
            isZeroPronoun: false,
          });
        }
      }

      if (args.length > 0) {
        results.push({
          predicateIndex: i,
          predicateText: predText,
          arguments: args,
        });
      }
    }

    return results;
  }
}
