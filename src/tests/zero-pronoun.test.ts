import { describe, it, expect, beforeEach } from 'vitest';
import { BiaffinePASHead, JAPANESE_PAS_CASES } from '../model/BiaffinePASHead.js';
import {
  ZeroPronounResolver,
  AntecedentCandidate,
  ContextSentence,
} from '../model/ZeroPronounResolver.js';

describe('ZeroPronounResolver', () => {
  let pasHead: BiaffinePASHead;
  let resolver: ZeroPronounResolver;

  beforeEach(() => {
    pasHead = new BiaffinePASHead();
    resolver = new ZeroPronounResolver(pasHead);
  });

  it('should initialize with default configuration and role weights', () => {
    expect(resolver.config.maxSentenceWindow).toBe(3);
    expect(resolver.config.distanceDecay).toBe(0.75);
    expect(resolver.config.confidenceThreshold).toBe(0.3);
    expect(resolver.config.mandatoryCases).toEqual(['ガ']);
    expect(resolver.config.roleWeights['ガ']).toBe(1.0);
    expect(resolver.config.roleWeights['ハ']).toBe(0.9);
  });

  it('should collect antecedent candidates strictly within 3 preceding sentences', () => {
    const s0: ContextSentence = {
      sentenceId: 's0',
      text: '昔々、太郎が住んでいた。',
      index: 0,
      entities: [
        { id: 'e_taro', text: '太郎', entityType: 'character', sentenceDistance: 0, caseRole: 'ガ' },
      ],
    };

    const s1: ContextSentence = {
      sentenceId: 's1',
      text: '花子は本を買った。',
      index: 1,
      entities: [
        { id: 'e_hanako', text: '花子', entityType: 'character', sentenceDistance: 0, caseRole: 'ガ' },
        { id: 'e_book', text: '本', entityType: 'noun', sentenceDistance: 0, caseRole: 'ヲ' },
      ],
    };

    const s2: ContextSentence = {
      sentenceId: 's2',
      text: '次郎が街を歩いた。',
      index: 2,
      entities: [
        { id: 'e_jiro', text: '次郎', entityType: 'character', sentenceDistance: 0, caseRole: 'ガ' },
      ],
    };

    const s3: ContextSentence = {
      sentenceId: 's3',
      text: '三郎が川に行った。',
      index: 3,
      entities: [
        { id: 'e_saburo', text: '三郎', entityType: 'character', sentenceDistance: 0, caseRole: 'ガ' },
      ],
    };

    const currentSentence: ContextSentence = {
      sentenceId: 's4',
      text: '微笑んだ。', // Omitted subject
      index: 4,
      entities: [],
    };

    // Sentences in context: s0 (index 0, dist 4), s1 (index 1, dist 3), s2 (index 2, dist 2), s3 (index 3, dist 1)
    const candidates = resolver.collectCandidatesFromContext(currentSentence, [s0, s1, s2, s3]);

    // s0 (dist 4 > maxSentenceWindow 3) should be excluded!
    const candidateIds = candidates.map((c) => c.id);
    expect(candidateIds).not.toContain('e_taro');
    expect(candidateIds).toContain('e_saburo'); // dist 1
    expect(candidateIds).toContain('e_jiro');   // dist 2
    expect(candidateIds).toContain('e_hanako'); // dist 3
    expect(candidateIds).toContain('e_book');   // dist 3

    // Check sentence distance mapping
    const saburoCand = candidates.find((c) => c.id === 'e_saburo');
    expect(saburoCand?.sentenceDistance).toBe(1);

    const hanakoCand = candidates.find((c) => c.id === 'e_hanako');
    expect(hanakoCand?.sentenceDistance).toBe(3);
  });

  it('should score candidates based on distance decay, syntactic prominence, and salience', () => {
    const candCloseSubject: AntecedentCandidate = {
      id: 'e1',
      text: '太郎',
      entityType: 'character',
      sentenceDistance: 1,
      caseRole: 'ガ', // Subject
      salienceScore: 1.0,
    };

    const candFarObject: AntecedentCandidate = {
      id: 'e2',
      text: '本',
      entityType: 'noun',
      sentenceDistance: 3,
      caseRole: 'ヲ', // Object
      salienceScore: 1.0,
    };

    const score1 = resolver.scoreCandidate(candCloseSubject, 'ガ', 0.8);
    const score2 = resolver.scoreCandidate(candFarObject, 'ガ', 0.2);

    expect(score1.totalScore).toBeGreaterThan(score2.totalScore);
    expect(score1.distanceScore).toBeGreaterThan(score2.distanceScore);
    expect(score1.syntacticProminence).toBeGreaterThan(score2.syntacticProminence);
  });

  it('should rank candidates and normalize anaphora likelihood using softmax', () => {
    const candidates: AntecedentCandidate[] = [
      { id: 'c1', text: '太郎', entityType: 'character', sentenceDistance: 1, caseRole: 'ガ' },
      { id: 'c2', text: '花子', entityType: 'character', sentenceDistance: 2, caseRole: 'ハ' },
      { id: 'c3', text: 'リンゴ', entityType: 'noun', sentenceDistance: 3, caseRole: 'ヲ' },
    ];

    const ranked = resolver.rankCandidates('ガ', 0, candidates, [0.9, 0.6, 0.1]);

    expect(ranked.length).toBe(3);
    // Highest total score first
    expect(ranked[0].candidate.id).toBe('c1');
    expect(ranked[1].candidate.id).toBe('c2');
    expect(ranked[2].candidate.id).toBe('c3');

    // Anaphora likelihoods sum to ~1.0
    const sumLikelihood = ranked.reduce((acc, r) => acc + r.anaphoraLikelihood, 0);
    expect(sumLikelihood).toBeCloseTo(1.0);
    expect(ranked[0].anaphoraLikelihood).toBeGreaterThan(ranked[1].anaphoraLikelihood);
  });

  it('should check mandatory slot fulfillment and normalize incomplete sentence scores', () => {
    const completeCheck = resolver.checkMandatorySlots('読んだ', ['ガ', 'ヲ']);
    expect(completeCheck.isComplete).toBe(true);
    expect(completeCheck.missingSlots).toEqual([]);

    const incompleteCheck = resolver.checkMandatorySlots('読んだ', ['ヲ']);
    expect(incompleteCheck.isComplete).toBe(false);
    expect(incompleteCheck.missingSlots).toEqual(['ガ']);

    const rawScore = 1.0;
    const normScoreComplete = resolver.normalizeIncompleteSentenceScore(rawScore, 0, 0.0);
    expect(normScoreComplete).toBe(1.0);

    const normScoreIncomplete = resolver.normalizeIncompleteSentenceScore(rawScore, 1, 0.8);
    expect(normScoreIncomplete).toBeLessThan(rawScore);
    expect(normScoreIncomplete).toBeCloseTo(1.0 * (1.0 - (0.2 * 1 + 0.3 * 0.8)));
  });

  it('should resolve zero pronouns from PAS relations and context', () => {
    const s1: ContextSentence = {
      sentenceId: 's1',
      text: '太郎は学校へ行った。',
      index: 1,
      entities: [
        { id: 'taro', text: '太郎', entityType: 'character', sentenceDistance: 0, caseRole: 'ハ' },
      ],
    };

    const currentSentence: ContextSentence = {
      sentenceId: 's2',
      text: '本を読んだ。',
      index: 2,
      entities: [],
      pasRelations: [
        {
          predicateIndex: 2,
          predicateText: '読んだ',
          arguments: [
            { caseName: 'ヲ', argIndex: 0, argText: '本', score: 0.9, isZeroPronoun: false },
            { caseName: 'ガ', argIndex: null, score: 0.85, isZeroPronoun: true }, // Zero pronoun subject
          ],
        },
      ],
    };

    const resolutions = resolver.resolveZeroPronouns(currentSentence, [s1]);

    expect(resolutions.length).toBe(1);
    const res = resolutions[0];
    expect(res.predicateIndex).toBe(2);
    expect(res.omittedCase).toBe('ガ');
    expect(res.isResolved).toBe(true);
    expect(res.bestCandidate?.candidate.id).toBe('taro');
  });

  it('should analyze incomplete sentence and produce full resolution metrics', () => {
    const s1: ContextSentence = {
      sentenceId: 's1',
      text: 'アリスは走っていた。',
      index: 1,
      entities: [
        { id: 'alice', text: 'アリス', entityType: 'character', sentenceDistance: 0, caseRole: 'ハ' },
      ],
    };

    const currentSentence: ContextSentence = {
      sentenceId: 's2',
      text: '転んだ。',
      index: 2,
      entities: [],
      predicates: [{ text: '転んだ', index: 0 }],
    };

    const analysis = resolver.analyzeIncompleteSentence(currentSentence, [s1], undefined, 0.95);

    expect(analysis.sentenceText).toBe('転んだ。');
    expect(analysis.isIncomplete).toBe(true);
    expect(analysis.missingMandatorySlots).toContain('ガ');
    expect(analysis.normalizedSentenceScore).toBeLessThan(analysis.rawSentenceScore);
    expect(analysis.resolutions.length).toBe(1);
    expect(analysis.resolutions[0].bestCandidate?.candidate.id).toBe('alice');
  });
});
