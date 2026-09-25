import {
  BiaffinePASHead,
  JAPANESE_PAS_CASES,
  PASCase,
  PASPredicateRelation,
  PASArgument,
} from './BiaffinePASHead.js';

export interface AntecedentCandidate {
  id: string;
  text: string;
  entityType?: 'character' | 'noun' | 'location' | string;
  sentenceDistance: number; // 0 = current sentence, 1 = 1 sentence ago, 2 = 2 sentences ago, 3 = 3 sentences ago
  caseRole?: PASCase | 'ハ' | string; // Syntactic role in its originating sentence
  salienceScore?: number; // Base entity salience / frequency (default: 1.0)
  tokenIndex?: number;
  vector?: Float32Array | number[];
}

export interface ContextSentence {
  sentenceId: string | number;
  text: string;
  index: number;
  entities: AntecedentCandidate[];
  predicates?: {
    text: string;
    index: number;
    mandatoryCases?: PASCase[];
  }[];
  pasRelations?: PASPredicateRelation[];
}

export interface ScoreBreakdown {
  pasScore: number;
  distanceScore: number;
  syntacticProminence: number;
  salienceScore: number;
}

export interface RankedCandidate {
  candidate: AntecedentCandidate;
  totalScore: number;
  anaphoraLikelihood: number; // Normalized probability [0..1]
  breakdown: ScoreBreakdown;
}

export interface ZeroPronounResolutionResult {
  predicateIndex: number;
  predicateText?: string;
  omittedCase: PASCase; // e.g. 'ガ'
  zeroPronounScore: number; // Confidence that case argument is omitted (from PAS head)
  rankedCandidates: RankedCandidate[];
  bestCandidate: RankedCandidate | null;
  isResolved: boolean;
}

export interface IncompleteSentenceAnalysis {
  sentenceText: string;
  isIncomplete: boolean;
  missingMandatorySlots: PASCase[];
  rawSentenceScore: number;
  normalizedSentenceScore: number;
  zeroPronounLikelihood: number;
  resolutions: ZeroPronounResolutionResult[];
}

export interface ZeroPronounResolverConfig {
  maxSentenceWindow?: number; // Max preceding sentences to search (default: 3)
  distanceDecay?: number; // Exponential decay factor per sentence distance (default: 0.75)
  confidenceThreshold?: number; // Zero pronoun detection score threshold (default: 0.3)
  resolutionThreshold?: number; // Min likelihood to declare resolved (default: 0.25)
  mandatoryCases?: PASCase[]; // Default required case slots for Japanese verbs (default: ['ガ'])
  roleWeights?: Record<string, number>; // Weights for syntactic roles
}

export class ZeroPronounResolver {
  public readonly config: Required<ZeroPronounResolverConfig>;
  private pasHead: BiaffinePASHead;

  constructor(
    pasHead?: BiaffinePASHead,
    config?: ZeroPronounResolverConfig
  ) {
    this.pasHead = pasHead ?? new BiaffinePASHead();
    this.config = {
      maxSentenceWindow: config?.maxSentenceWindow ?? 3,
      distanceDecay: config?.distanceDecay ?? 0.75,
      confidenceThreshold: config?.confidenceThreshold ?? 0.3,
      resolutionThreshold: config?.resolutionThreshold ?? 0.25,
      mandatoryCases: config?.mandatoryCases ?? ['ガ'],
      roleWeights: {
        'ガ': 1.0,  // Subject case - highest prominence
        'ガ２': 0.95,
        'ハ': 0.9,   // Topic marker - high prominence
        'ヲ': 0.65,  // Direct object
        'ニ': 0.55,  // Indirect object
        'ト': 0.45,
        'デ': 0.4,
        'カラ': 0.35,
        'ヘ': 0.35,
        'ヨリ': 0.3,
        'マデ': 0.3,
        ...(config?.roleWeights ?? {}),
      },
    };
  }

  /**
   * Checks mandatory slot fulfillment for a predicate given explicit cases
   */
  public checkMandatorySlots(
    predicateText: string,
    filledCases: PASCase[],
    customMandatoryCases?: PASCase[]
  ): { isComplete: boolean; missingSlots: PASCase[] } {
    const required = customMandatoryCases ?? this.config.mandatoryCases;
    const missingSlots = required.filter((c) => !filledCases.includes(c));
    return {
      isComplete: missingSlots.length === 0,
      missingSlots,
    };
  }

  /**
   * Normalizes score for an incomplete sentence with missing mandatory slots
   */
  public normalizeIncompleteSentenceScore(
    rawScore: number,
    missingSlotsCount: number,
    zeroPronounLikelihood: number
  ): number {
    if (missingSlotsCount === 0) {
      return rawScore;
    }

    // Penalty scales with number of missing mandatory slots and zero-pronoun confidence
    const slotPenalty = 0.2 * missingSlotsCount;
    const zpPenalty = 0.3 * zeroPronounLikelihood;
    const factor = Math.max(0.1, 1.0 - (slotPenalty + zpPenalty));

    return rawScore * factor;
  }

  /**
   * Collect antecedent candidates from context within max 3 preceding sentences
   */
  public collectCandidatesFromContext(
    currentSentence: ContextSentence,
    recentContext: ContextSentence[]
  ): AntecedentCandidate[] {
    const window = this.config.maxSentenceWindow; // Max 3 sentences
    const candidates: AntecedentCandidate[] = [];
    const seenIds = new Set<string>();

    // 1. Current sentence entities (sentenceDistance = 0)
    for (const ent of currentSentence.entities) {
      if (!seenIds.has(ent.id)) {
        seenIds.add(ent.id);
        candidates.push({
          ...ent,
          sentenceDistance: 0,
        });
      }
    }

    // 2. Recent preceding context sentences (within window 1..3)
    // Filter context sentences to those within recent 3 sentences
    const recentSentences = recentContext
      .filter((s) => s.index < currentSentence.index && currentSentence.index - s.index <= window)
      .sort((a, b) => b.index - a.index); // Closest first

    for (const sent of recentSentences) {
      const distance = currentSentence.index - sent.index;
      if (distance > window) continue;

      for (const ent of sent.entities) {
        if (!seenIds.has(ent.id)) {
          seenIds.add(ent.id);
          candidates.push({
            ...ent,
            sentenceDistance: distance,
          });
        }
      }
    }

    return candidates;
  }

  /**
   * Calculate score and anaphora likelihood for candidate entity
   */
  public scoreCandidate(
    candidate: AntecedentCandidate,
    omittedCase: PASCase,
    pasScore: number = 0.5
  ): ScoreBreakdown & { totalScore: number } {
    // 1. Distance score with exponential decay
    const distanceScore = Math.pow(this.config.distanceDecay, candidate.sentenceDistance);

    // 2. Syntactic prominence based on previous case role or topic marker
    const roleKey = candidate.caseRole ?? 'default';
    const syntacticProminence = this.config.roleWeights[roleKey] ?? 0.3;

    // 3. Entity salience score (default 1.0, boosted for character types)
    let salienceScore = candidate.salienceScore ?? 1.0;
    if (candidate.entityType === 'character') {
      salienceScore *= 1.2;
    }

    // 4. Combined total score
    // totalScore = pasScore * 0.35 + distanceScore * 0.30 + syntacticProminence * 0.20 + salienceScore * 0.15
    const totalScore =
      pasScore * 0.35 +
      distanceScore * 0.30 +
      syntacticProminence * 0.20 +
      Math.min(salienceScore, 2.0) * 0.15;

    return {
      pasScore,
      distanceScore,
      syntacticProminence,
      salienceScore,
      totalScore,
    };
  }

  /**
   * Ranks candidates for an omitted case in a predicate using softmax likelihood normalization
   */
  public rankCandidates(
    omittedCase: PASCase,
    predicateIndex: number,
    candidates: AntecedentCandidate[],
    pasScoresForCandidates?: number[]
  ): RankedCandidate[] {
    if (candidates.length === 0) return [];

    const scoredList = candidates.map((cand, idx) => {
      const pasVal = pasScoresForCandidates?.[idx] ?? 0.5;
      const breakdown = this.scoreCandidate(cand, omittedCase, pasVal);
      return {
        candidate: cand,
        breakdown,
        totalScore: breakdown.totalScore,
      };
    });

    // Softmax normalization for Anaphora Likelihood
    const maxScore = Math.max(...scoredList.map((s) => s.totalScore));
    const expScores = scoredList.map((s) => Math.exp((s.totalScore - maxScore) * 2.0)); // temperature = 0.5
    const sumExp = expScores.reduce((a, b) => a + b, 0);

    const ranked: RankedCandidate[] = scoredList.map((s, idx) => ({
      candidate: s.candidate,
      totalScore: s.totalScore,
      anaphoraLikelihood: expScores[idx] / Math.max(sumExp, 1e-9),
      breakdown: {
        pasScore: s.breakdown.pasScore,
        distanceScore: s.breakdown.distanceScore,
        syntacticProminence: s.breakdown.syntacticProminence,
        salienceScore: s.breakdown.salienceScore,
      },
    }));

    // Sort descending by totalScore / anaphoraLikelihood
    return ranked.sort((a, b) => b.totalScore - a.totalScore);
  }

  /**
   * Resolves zero pronouns for a given sentence and context using PAS head scores
   */
  public resolveZeroPronouns(
    currentSentence: ContextSentence,
    recentContext: ContextSentence[],
    rawPasScores?: number[][][] | Float32Array
  ): ZeroPronounResolutionResult[] {
    const candidates = this.collectCandidatesFromContext(currentSentence, recentContext);
    const results: ZeroPronounResolutionResult[] = [];

    let pasRelations = currentSentence.pasRelations;

    // If raw PAS tensor is supplied, extract relations using BiaffinePASHead
    if (rawPasScores && (!pasRelations || pasRelations.length === 0)) {
      pasRelations = this.pasHead.extractRelations(rawPasScores, {
        sequenceText: currentSentence.text,
        threshold: 0.15,
        zeroPronounThreshold: this.config.confidenceThreshold,
      });
    }

    if (!pasRelations || pasRelations.length === 0) {
      // Fallback: check predicate descriptors if available
      if (currentSentence.predicates) {
        for (const pred of currentSentence.predicates) {
          const omittedCase: PASCase = 'ガ'; // Subject zero pronoun check
          const ranked = this.rankCandidates(omittedCase, pred.index, candidates);
          const best = ranked.length > 0 && ranked[0].anaphoraLikelihood >= this.config.resolutionThreshold
            ? ranked[0]
            : null;

          results.push({
            predicateIndex: pred.index,
            predicateText: pred.text,
            omittedCase,
            zeroPronounScore: 0.8, // Default confidence for incomplete predicate
            rankedCandidates: ranked,
            bestCandidate: best,
            isResolved: best !== null,
          });
        }
      }
      return results;
    }

    for (const rel of pasRelations) {
      for (const arg of rel.arguments) {
        if (arg.isZeroPronoun && (arg.caseName === 'ガ' || arg.caseName === 'ガ２')) {
          const omittedCase = arg.caseName as PASCase;
          const ranked = this.rankCandidates(omittedCase, rel.predicateIndex, candidates);
          const best = ranked.length > 0 && ranked[0].anaphoraLikelihood >= this.config.resolutionThreshold
            ? ranked[0]
            : null;

          results.push({
            predicateIndex: rel.predicateIndex,
            predicateText: rel.predicateText,
            omittedCase,
            zeroPronounScore: arg.score,
            rankedCandidates: ranked,
            bestCandidate: best,
            isResolved: best !== null,
          });
        }
      }
    }

    return results;
  }

  /**
   * Analyze an incomplete sentence for missing mandatory slots and zero pronoun resolution
   */
  public analyzeIncompleteSentence(
    currentSentence: ContextSentence,
    recentContext: ContextSentence[],
    rawPasScores?: number[][][] | Float32Array,
    rawSentenceScore: number = 1.0
  ): IncompleteSentenceAnalysis {
    const resolutions = this.resolveZeroPronouns(currentSentence, recentContext, rawPasScores);

    // Collect all explicit filled cases in current sentence
    const filledCases: PASCase[] = [];
    if (currentSentence.pasRelations) {
      for (const rel of currentSentence.pasRelations) {
        for (const arg of rel.arguments) {
          if (!arg.isZeroPronoun && arg.argIndex !== null) {
            filledCases.push(arg.caseName as PASCase);
          }
        }
      }
    }

    const { isComplete, missingSlots } = this.checkMandatorySlots(
      currentSentence.text,
      filledCases
    );

    // Max zero pronoun likelihood across resolutions
    const maxZpScore = resolutions.length > 0
      ? Math.max(...resolutions.map((r) => r.zeroPronounScore))
      : (missingSlots.length > 0 ? 0.8 : 0.0);

    const isIncomplete = !isComplete || resolutions.length > 0;
    const normalizedSentenceScore = this.normalizeIncompleteSentenceScore(
      rawSentenceScore,
      missingSlots.length,
      maxZpScore
    );

    return {
      sentenceText: currentSentence.text,
      isIncomplete,
      missingMandatorySlots: missingSlots,
      rawSentenceScore,
      normalizedSentenceScore,
      zeroPronounLikelihood: maxZpScore,
      resolutions,
    };
  }
}
