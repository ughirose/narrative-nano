import { describe, it, expect } from 'vitest';
import { EpistemicPOVDetector, type POVSentenceInput } from '../model/EpistemicPOVDetector.js';

describe('EpistemicPOVDetector', () => {
  it('should detect unauthorized internal state in first_person POV', () => {
    const detector = new EpistemicPOVDetector({
      mode: 'first_person',
      narratorPronouns: ['僕', '私'],
    });

    const sentences: POVSentenceInput[] = [
      { index: 0, text: '僕は雨の降る街を歩いていた。', isDialogue: false, subjectCandidate: '僕' },
      { index: 1, text: '冷たい雫が頬を伝い、寂しさを感じた。', isDialogue: false, subjectCandidate: '僕' },
      { index: 2, text: '通りすがりの男は、心の中で僕を嘲笑った。', isDialogue: false, subjectCandidate: '男' },
      { index: 3, text: '「おい、邪魔だ」', isDialogue: true },
    ];

    const report = detector.analyzeSentences(sentences);
    expect(report.violations.length).toBe(1);
    expect(report.violations[0].violationType).toBe('unauthorized_internal_state');
    expect(report.violations[0].culpritEntity).toBe('男');
    expect(report.violations[0].sentenceIndex).toBe(2);
    expect(report.violations[0].suggestion).toBeDefined();
  });

  it('should allow internal state with conjectural mitigation (推量表現)', () => {
    const detector = new EpistemicPOVDetector({
      mode: 'first_person',
      narratorPronouns: ['私'],
    });

    const sentences: POVSentenceInput[] = [
      { index: 0, text: '彼は悔しそうに見えた。', isDialogue: false, subjectCandidate: '彼' },
      { index: 1, text: '相手は恐れているようだった。', isDialogue: false, subjectCandidate: '相手' },
      { index: 2, text: '彼女は私を軽蔑しているに違いない。', isDialogue: false, subjectCandidate: '彼女' },
    ];

    const report = detector.analyzeSentences(sentences);
    expect(report.violations.length).toBe(0);
    expect(report.povConsistencyScore).toBe(1.0);
  });

  it('should detect head-hopping in third_person_limited POV', () => {
    const detector = new EpistemicPOVDetector({
      mode: 'third_person_limited',
      povCharacter: 'アリス',
    });

    const sentences: POVSentenceInput[] = [
      { index: 0, text: 'アリスは暗い森の奥を見つめ、恐怖した。', isDialogue: false, subjectCandidate: 'アリス' },
      { index: 1, text: 'ボブは心の内で勝機を見出し、ほっとした。', isDialogue: false, subjectCandidate: 'ボブ' },
    ];

    const report = detector.analyzeSentences(sentences);
    expect(report.violations.length).toBeGreaterThanOrEqual(1);
    const unauthorized = report.violations.find((v) => v.violationType === 'unauthorized_internal_state');
    expect(unauthorized).toBeDefined();
    expect(unauthorized?.culpritEntity).toBe('ボブ');
  });

  it('should strictly prohibit all internal states in third_person_objective (camera eye) mode', () => {
    const detector = new EpistemicPOVDetector({
      mode: 'third_person_objective',
    });

    const sentences: POVSentenceInput[] = [
      { index: 0, text: '部屋の中には二人の男が座っていた。', isDialogue: false },
      { index: 1, text: '警官は後悔した。', isDialogue: false, subjectCandidate: '警官' },
    ];

    const report = detector.analyzeSentences(sentences);
    expect(report.violations.length).toBe(1);
    expect(report.violations[0].violationType).toBe('objective_mode_leak');
    expect(report.violations[0].severity).toBe('error');
  });
});
