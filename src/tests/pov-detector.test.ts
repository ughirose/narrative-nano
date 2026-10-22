import { describe, it, expect } from 'vitest';
import { EpistemicPOVDetector, type POVSentenceInput } from '../model/EpistemicPOVDetector.js';
import { ContextDisambiguator } from '../model/ContextDisambiguator.js';

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

  describe('ContextDisambiguator with Kasane Telemetry', () => {
    it('should adapt third-person narrative with confession/tension spike into Monologue (FID)', () => {
      const text = '彼女は立ち尽くした。なぜこんな仕打ちを受けねばならないのか。';
      const rawPreds = new Array(text.length).fill(0);
      const telemetry = {
        texture: {
          showing: 30.0,
          staccato: 40.0,
          polyphony: 40.0,
          confession: 78.0,
          tension: 0.65,
        },
        dynamics: {
          crestFactor: 1.2,
        },
      };

      const refined = ContextDisambiguator.refine(text, rawPreds, { telemetry });

      // 前半「彼女は立ち尽くした。」(10文字) は 0
      expect(refined.slice(0, 10).every((c) => c === 0)).toBe(true);
      // 後半「なぜこんな仕打ちを受けねばならないのか。」(20文字) は 2 (Monologue / FID)
      expect(refined.slice(10).every((c) => c === 2)).toBe(true);
    });

    it('should suppress FID false positives in past recollection context', () => {
      const text = '彼女は立ち尽くした。なぜこんな仕打ちを受けねばならないのか。';
      const rawPreds = new Array(text.length).fill(0);
      const telemetry = {
        texture: {
          showing: 30.0,
          staccato: 40.0,
          polyphony: 40.0,
          confession: 78.0, // 80.0 未満
          tension: 0.65,
        },
        dynamics: {
          crestFactor: 1.2,
        },
      };

      // 過去回想コンテキストでは閾値が 80.0 に引き上げられ、78.0 では補正されない
      const refined = ContextDisambiguator.refine(text, rawPreds, {
        telemetry,
        isPastRecollectionContext: true,
      });

      expect(refined.every((c) => c === 0)).toBe(true);
    });

    it('should modulate modality probability distribution using Bayesian prior', () => {
      const telemetry = {
        texture: {
          showing: 30.0,
          staccato: 40.0,
          polyphony: 40.0,
          confession: 80.0,
          tension: 0.7,
        },
        dynamics: {
          crestFactor: 1.2,
        },
      };

      const modulated = ContextDisambiguator.modulateModalityProbabilities(0.7, 0.3, telemetry, 0.5);
      // P*(Monologue) = 0.3 * (1.0 + 0.5 * 0.8) = 0.3 * 1.4 = 0.42
      // P*(Narrative) = 1.0 - 0.42 = 0.58
      expect(modulated.monologue).toBeCloseTo(0.42);
      expect(modulated.narrative).toBeCloseTo(0.58);
    });
  });

  describe('EpistemicPOVDetector with Kasane Telemetry', () => {
    it('should trigger excessive speculation penalty when showing is low and speculation is high', () => {
      const detector = new EpistemicPOVDetector({
        mode: 'first_person',
        narratorPronouns: ['私'],
      });

      const sentences: POVSentenceInput[] = [
        { index: 0, text: '彼女は私を軽蔑しているに違いない。', isDialogue: false, subjectCandidate: '彼女' },
      ];

      // 通常時: 推量免除によりスルー
      const normalReport = detector.analyzeSentences(sentences);
      expect(normalReport.violations.length).toBe(0);

      // テレメトリ注入時: showing: 15.0, speculation: 74.0, staccato: 18.0
      const telemetry = {
        texture: {
          showing: 15.0,
          speculation: 74.0,
          staccato: 18.0,
          polyphony: 40.0,
          confession: 30.0,
          tension: 0.2,
        },
        dynamics: {
          crestFactor: 1.0,
        },
      };

      const penalizedReport = detector.analyzeSentences(sentences, telemetry);
      expect(penalizedReport.violations.length).toBe(1);
      const v = penalizedReport.violations[0];
      expect(v.violationType).toBe('epistemic_leak_excessive_speculation');
      expect(v.culpritEntity).toBe('彼女');
      expect(v.message).toContain('【説明過多・認知漏洩の兆候】');
      expect(v.message).toContain('Showing: 15.0');
      expect(v.message).toContain('Speculation: 74.0');
      expect(v.confidence).toBeGreaterThan(0.85);
    });

    it('should detect epistemic leak via 5-sentence sliding window local speculation spike', () => {
      const detector = new EpistemicPOVDetector({
        mode: 'first_person',
        narratorPronouns: ['私'],
      });

      const baseTelemetry = (spec: number) => ({
        texture: {
          showing: 30.0,
          speculation: spec,
          staccato: 40.0,
          polyphony: 30.0,
          confession: 20.0,
          tension: 0.1,
        },
        dynamics: { crestFactor: 1.0 },
      });

      const sentences: POVSentenceInput[] = [
        { index: 0, text: '街は静まり返っていた。', isDialogue: false, kasaneTelemetry: baseTelemetry(15.0) },
        { index: 1, text: '風が吹き抜けていく。', isDialogue: false, kasaneTelemetry: baseTelemetry(20.0) },
        { index: 2, text: '街灯がかすかに揺れた。', isDialogue: false, kasaneTelemetry: baseTelemetry(18.0) },
        { index: 3, text: '足音が遠くで響く。', isDialogue: false, kasaneTelemetry: baseTelemetry(22.0) },
        // 5文目で speculation が 75.0 へ急騰スパイク（直前窓 min 15.0 から +60.0 スパイク）
        {
          index: 4,
          text: '彼は私を恨んでいるに違いない。',
          isDialogue: false,
          subjectCandidate: '彼',
          kasaneTelemetry: baseTelemetry(75.0),
        },
      ];

      const report = detector.analyzeSentences(sentences);
      expect(report.violations.length).toBe(1);
      expect(report.violations[0].sentenceIndex).toBe(4);
      expect(report.violations[0].violationType).toBe('epistemic_leak_excessive_speculation');
      expect(report.violations[0].culpritEntity).toBe('彼');
      expect(report.violations[0].message).toContain('【説明過多・認知漏洩の兆候】');
    });

    it('should soft-land Free Indirect Discourse (FID) to severity: info under confession and tension modulation', () => {
      const detector = new EpistemicPOVDetector({
        mode: 'third_person_limited',
        povCharacter: 'アリス',
      });

      const sentences: POVSentenceInput[] = [
        { index: 0, text: 'アリスは立ち尽くした。', isDialogue: false, subjectCandidate: 'アリス' },
        { index: 1, text: 'なぜこんな仕打ちを受けねばならないのか。', isDialogue: false },
      ];

      const fidTelemetry = {
        texture: {
          showing: 35.0,
          staccato: 30.0,
          polyphony: 40.0,
          confession: 78.0,
          tension: 0.65,
        },
        dynamics: { crestFactor: 1.2 },
      };

      const report = detector.analyzeSentences(sentences, fidTelemetry);
      expect(report.violations.length).toBe(1);
      const fidViolation = report.violations[0];
      expect(fidViolation.violationType).toBe('free_indirect_discourse');
      expect(fidViolation.severity).toBe('info');
      expect(fidViolation.sentenceIndex).toBe(1);
      expect(fidViolation.message).toContain('自由間接話法（FID: 内面浸透）');
      // severity: info のため減点ペナルティが 0 で一貫性スコアは 1.0 のまま
      expect(report.povConsistencyScore).toBe(1.0);
    });
  });
});
