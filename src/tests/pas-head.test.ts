import { describe, it, expect, beforeEach } from 'vitest';
import {
  BiaffinePASHead,
  JAPANESE_PAS_CASES,
  PASCase,
} from '../model/BiaffinePASHead';

describe('BiaffinePASHead', () => {
  it('should initialize with default parameters (256 hidden dim, 10 Japanese cases)', () => {
    const head = new BiaffinePASHead();
    expect(head.config.hiddenDim).toBe(256);
    expect(head.config.numCases).toBe(10);
    expect(head.config.cases).toEqual(JAPANESE_PAS_CASES);

    const weights = head.getWeights();
    expect(weights.wPred.length).toBe(256 * 256);
    expect(weights.wArg.length).toBe(256 * 256);
    expect(weights.uCases.length).toBe(10 * 256 * 256);
  });

  it('should allow setting custom weights and configuration', () => {
    const head = new BiaffinePASHead({ hiddenDim: 64, numCases: 10 });
    expect(head.config.hiddenDim).toBe(64);

    const d = 64;
    const c = 10;
    const customWPred = new Float32Array(d * d).fill(0.1);
    const customWArg = new Float32Array(d * d).fill(0.2);
    const customUCases = new Float32Array(c * d * d).fill(0.05);

    head.setWeights({
      wPred: customWPred,
      wArg: customWArg,
      uCases: customUCases,
    });

    const w = head.getWeights();
    expect(w.wPred[0]).toBeCloseTo(0.1);
    expect(w.wArg[0]).toBeCloseTo(0.2);
    expect(w.uCases[0]).toBeCloseTo(0.05);
  });

  it('should compute forward bilinear inner-product tensor correctly', () => {
    const hiddenDim = 4;
    const numCases = 2;
    const cases = ['ガ', 'ヲ'];

    const head = new BiaffinePASHead({ hiddenDim, numCases, cases });

    // Set identity-like weights for predictable arithmetic
    const wPred = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
    const wArg = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
    // U[0] = Identity, U[1] = 2 * Identity
    const uCases = new Float32Array(2 * 4 * 4);
    for (let i = 0; i < 4; i++) {
      uCases[0 * 16 + i * 4 + i] = 1.0;
      uCases[1 * 16 + i * 4 + i] = 2.0;
    }

    head.setWeights({ wPred, wArg, uCases });

    // Sequence length = 2
    const hiddenStates = [
      [1, 0, 0, 0], // token 0
      [0, 1, 0, 0], // token 1
    ];

    const scores = head.forward(hiddenStates);

    expect(scores.length).toBe(2);    // 2 cases
    expect(scores[0].length).toBe(2); // L = 2 predicates
    expect(scores[0][0].length).toBe(2); // L = 2 arguments

    // S[case=0][i=0][j=0] = [1,0,0,0] . I . [1,0,0,0]^T = 1.0
    expect(scores[0][0][0]).toBeCloseTo(1.0);
    // S[case=0][i=0][j=1] = [1,0,0,0] . I . [0,1,0,0]^T = 0.0
    expect(scores[0][0][1]).toBeCloseTo(0.0);
    // S[case=1][i=0][j=0] = [1,0,0,0] . (2*I) . [1,0,0,0]^T = 2.0
    expect(scores[1][0][0]).toBeCloseTo(2.0);
  });

  it('should correctly parse Wasm SIMD / ONNX Runtime flat output tensor', () => {
    const head = new BiaffinePASHead({ hiddenDim: 256, numCases: 10 });
    const seqLen = 3;
    const numCases = 10;
    const flatLen = numCases * seqLen * seqLen;

    const flatBuffer = new Float32Array(flatLen);
    for (let i = 0; i < flatLen; i++) {
      flatBuffer[i] = i * 0.5;
    }

    const parsed = head.parseRawTensor(flatBuffer, seqLen, numCases);

    expect(parsed.length).toBe(10);
    expect(parsed[0].length).toBe(3);
    expect(parsed[0][0].length).toBe(3);

    // flatIdx for c=1, i=1, j=2 => 1 * 9 + 1 * 3 + 2 = 14
    expect(parsed[1][1][2]).toBeCloseTo(14 * 0.5);
  });

  it('should normalize scores and compute zero-pronoun candidate scores', () => {
    const head = new BiaffinePASHead();
    const seqLen = 3;
    const numCases = 10;

    // Create mock scores where token 0 for case 'ガ' has low scores for all args
    const rawScores: number[][][] = Array.from({ length: numCases }, () =>
      Array.from({ length: seqLen }, () => [0.1, 0.2, 0.1])
    );

    const { normalizedScores, zeroPronounScores } = head.normalizeScores(rawScores, seqLen, {
      temperature: 1.0,
      applySoftmax: true,
    });

    expect(normalizedScores.length).toBe(10);
    expect(zeroPronounScores.length).toBe(10);

    // Softmax probabilities should sum to ~1.0 for each row
    const rowSum = normalizedScores[0][0].reduce((a, b) => a + b, 0);
    expect(rowSum).toBeCloseTo(1.0);

    // Since max probability in softmax for [0.1, 0.2, 0.1] is ~0.366, zero pronoun score = 1 - 0.366 > 0.6
    expect(zeroPronounScores[0][0]).toBeGreaterThan(0.5);
  });

  it('should extract Japanese predicate-argument structures and handle zero pronouns', () => {
    const head = new BiaffinePASHead();
    const text = '本を読んだ。';
    const seqLen = text.length; // 6 chars: 本(0) を(1) 読(2) ん(3) だ(4) 。(5)

    const rawScores: number[][][] = Array.from({ length: 10 }, () =>
      Array.from({ length: seqLen }, () => new Array(seqLen).fill(-2.0))
    );

    // Set high score for predicate '読' (idx 2) and argument '本' (idx 0) for case 'ヲ' (idx 2)
    const oCaseIdx = JAPANESE_PAS_CASES.indexOf('ヲ');
    rawScores[oCaseIdx][2][0] = 5.0; // 本 -> ヲ -> 読んだ

    // Case 'ガ' (idx 0) has no explicit subject in text (zero pronoun)
    const gaCaseIdx = JAPANESE_PAS_CASES.indexOf('ガ');
    // rawScores[gaCaseIdx][2] stays low (-2.0)

    const relations = head.extractRelations(rawScores, {
      sequenceText: text,
      predicateIndices: [2], // '読'
      threshold: 0.15,
      zeroPronounThreshold: 0.3,
    });

    expect(relations.length).toBe(1);
    const rel = relations[0];
    expect(rel.predicateIndex).toBe(2);
    expect(rel.predicateText).toBe('読んだ。');

    const gaArg = rel.arguments.find((a) => a.caseName === 'ガ');
    expect(gaArg).toBeDefined();
    expect(gaArg?.isZeroPronoun).toBe(true);
    expect(gaArg?.argIndex).toBeNull();

    const oArg = rel.arguments.find((a) => a.caseName === 'ヲ');
    expect(oArg).toBeDefined();
    expect(oArg?.isZeroPronoun).toBe(false);
    expect(oArg?.argIndex).toBe(0);
    expect(oArg?.argText).toBe('本');
  });

  it('should penalize subject-predicate distortion candidate scores when distance is large', () => {
    const head = new BiaffinePASHead();
    const seqLen = 40; // sequence > 32 to trigger distortion check
    const rawScores: number[][][] = Array.from({ length: 10 }, () =>
      Array.from({ length: seqLen }, () => new Array(seqLen).fill(1.0))
    );

    const gaCaseIdx = JAPANESE_PAS_CASES.indexOf('ガ');
    const { normalizedScores } = head.normalizeScores(rawScores, seqLen, {
      distortionPenalty: 0.2,
    });

    // Score for near argument (j=2, dist=2) vs distant argument (j=38, dist=38) for predicate i=0
    expect(normalizedScores[gaCaseIdx][0][2]).toBeGreaterThan(normalizedScores[gaCaseIdx][0][38]);
  });
});
