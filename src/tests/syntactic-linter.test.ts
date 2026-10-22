import { describe, it, expect } from 'vitest';
import { SyntacticLinterRules } from '../linter/SyntacticLinterRules.js';

describe('SyntacticLinterRules', () => {
  describe('Particle Repetition Detection (助詞重複検知)', () => {
    it('should detect 3 or more occurrences of particle "が" in a single sentence', () => {
      const text = '彼が猫が魚が好きだと言った。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      expect(diagnostics.length).toBeGreaterThanOrEqual(3);
      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition'
      );
      expect(particleDiags.length).toBe(3);
      expect(particleDiags[0].message).toContain('助詞「が」が3回重複して使用されています');
      expect(particleDiags[0].severity).toBe('warning');
    });

    it('should detect 3 or more occurrences of particle "を" in a single sentence', () => {
      const text = '彼を部屋を荷物を見せに連れて行った。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition'
      );
      expect(particleDiags.length).toBe(3);
      expect(particleDiags[0].message).toContain('助詞「を」が3回重複して使用されています');
    });

    it('should detect 3 or more occurrences of particle "の" in a single sentence', () => {
      const text = '私の友達の犬の家の庭。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition'
      );
      expect(particleDiags.length).toBe(4);
      expect(particleDiags[0].message).toContain('助詞「の」が4回重複して使用されています');
    });

    it('should NOT flag particle repetitions if count is less than 3 (default)', () => {
      const text = '彼が本を読む。私が出かける。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition'
      );
      expect(particleDiags.length).toBe(0);
    });

    it('should NOT false-positive on particle "と" for adverbs (ひょっとすると), quotes (「と」), and compound particles (として)', () => {
      const text = '今わかったこととしては、しょうがないとは思うが。これはひょっとすると父親を意味する「とと」でも反応しそう。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition' && d.message.includes('「と」')
      );
      expect(particleDiags.length).toBe(0);
    });

    it('should support custom minParticleRepetitionCount option', () => {
      const text = '彼が本を読む。私が出かける。';
      const sentence = '彼が私が走る。';
      const diagnostics = SyntacticLinterRules.analyze(sentence, {
        minParticleRepetitionCount: 2,
      });

      const particleDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:particle-repetition'
      );
      expect(particleDiags.length).toBe(2);
      expect(particleDiags[0].message).toContain('助詞「が」が2回重複');
    });
  });

  describe('Subject-Predicate Mismatch & Twist Detection (主述不整合・ねじれ検知)', () => {
    it('should detect subject-predicate mismatch pattern "〜は、……と思ったからです。"', () => {
      const text = '私は、明日晴れると良いなと思ったからです。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const spDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:subject-predicate-mismatch'
      );
      expect(spDiags.length).toBe(1);
      expect(spDiags[0].severity).toBe('warning');
      expect(spDiags[0].message).toContain('主語と述語の文末係り受けのねじれ（主述不整合）を検知しました');
    });

    it('should detect subject mismatch with reason ending e.g. "私の夢は……優勝したからです。"', () => {
      const text = '私の夢は、幼い頃から努力を重ねて世界大会で優勝したからです。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const spDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:subject-predicate-mismatch'
      );
      expect(spDiags.length).toBe(1);
      expect(spDiags[0].message).toContain('主述不整合');
    });

    it('should NOT flag clean subject-predicate sentence', () => {
      const text = '私の夢は、世界大会で優勝することです。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      expect(diagnostics.length).toBe(0);
    });
  });

  describe('Clean Sentences & Valid Text', () => {
    it('should return 0 diagnostics for well-formed Japanese prose', () => {
      const text = '彼は静かに部屋の窓を開けた。外は穏やかな秋晴れで、心地よい風が吹き抜けた。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      expect(diagnostics.length).toBe(0);
    });
  });

  describe('Syntactic Score Calculation (構文スコア計算)', () => {
    it('should return high score (~1.0) for well-formed sentences', () => {
      const sentence = '彼は静かに本を読んだ。';
      const score = SyntacticLinterRules.calculateSyntacticScore(sentence);

      expect(score).toBeGreaterThanOrEqual(0.9);
    });

    it('should lower syntactic score when subject-predicate mismatch or particle repetition is present', () => {
      const badSentence = '私の夢は、努力を重ねて世界大会で優勝したからです。';
      const score = SyntacticLinterRules.calculateSyntacticScore(badSentence);

      expect(score).toBeLessThan(0.6);
    });
  });

  describe('CodeMirror 6 Linter Extension Integration', () => {
    it('should create CodeMirror 6 compatible linter function', () => {
      const linterFn = SyntacticLinterRules.createLinterExtension();

      const mockEditorView = {
        state: {
          doc: {
            toString: () => '彼が猫が魚が好きだと言った。',
          },
        },
      };

      const diagnostics = linterFn(mockEditorView);
      expect(Array.isArray(diagnostics)).toBe(true);
      expect(diagnostics.length).toBeGreaterThan(0);
      expect(diagnostics[0]).toHaveProperty('from');
      expect(diagnostics[0]).toHaveProperty('to');
      expect(diagnostics[0]).toHaveProperty('severity');
      expect(diagnostics[0]).toHaveProperty('message');
    });
  });

  describe('Double Negation Detection (二重否定検知)', () => {
    it('should detect double negation patterns e.g. "ないわけではない"', () => {
      const text = '彼が真相を知らないわけではない。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const dnDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:double-negation'
      );
      expect(dnDiags.length).toBe(1);
      expect(dnDiags[0].message).toContain('二重否定「ないわけではない」が検出されました');
    });

    it('should detect multiple double negation phrases e.g. "ざるを得ない", "なくもない"', () => {
      const text = '受け入れざるを得ないし、理解できなくもない。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const dnDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:double-negation'
      );
      expect(dnDiags.length).toBe(2);
    });
  });

  describe('Consecutive Passive Detection (受身の連続検知)', () => {
    it('should detect 2 or more passive forms in a single sentence', () => {
      const text = '敵に城を奪われて、味方が皆殺害された。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const cpDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:consecutive-passive'
      );
      expect(cpDiags.length).toBe(2);
      expect(cpDiags[0].message).toContain('同一文内で受身表現（計2箇所）が連続しています');
    });

    it('should NOT flag single passive form in a sentence', () => {
      const text = '彼は敵に追われた。';
      const diagnostics = SyntacticLinterRules.analyze(text);

      const cpDiags = diagnostics.filter((d) =>
        d.source === 'narrative-nano-linter:consecutive-passive'
      );
      expect(cpDiags.length).toBe(0);
    });
  });

  describe('Kasane Stylistic Telemetry & Intentional Irregularity (§3.3, §4.1, §4.3)', () => {
    it('should relax particle repetition threshold for classical/cadenced prose (cadence >= 30.0)', () => {
      const text = '彼が猫が魚が好きだと言った。'; // 助詞「が」が3回

      // 通常時: 警告が発報される
      const normalDiags = SyntacticLinterRules.analyze(text);
      expect(normalDiags.some((d) => d.message.includes('助詞「が」が3回重複'))).toBe(true);

      // 古典・高律格文体（cadence = 40.0 >= 30.0）: 閾値が 5 に緩和され、3回では警告されない
      const relaxedDiags = SyntacticLinterRules.analyze(text, {
        kasaneTelemetry: {
          texture: {
            showing: 50,
            staccato: 40,
            polyphony: 30,
            confession: 20,
            tension: 0.1,
            cadence: 40.0,
          },
          dynamics: { crestFactor: 1.0 },
        },
      });
      expect(relaxedDiags.filter((d) => d.source === 'narrative-nano-linter:particle-repetition').length).toBe(0);
    });

    it('should soft-land long-distance dependency to intentional irregularity (破格) under complex rhythm style', () => {
      const sentence = 'その古びた館の主人は、長年の沈黙を破り、ついに真相を語り始めた。';
      // PAS関係: 主語「主人は」(idx 8) と述語「語り始めた」(idx 25) -> 距離 17 > 12
      const pasRelations = [
        {
          predicateIndex: 25,
          predicateText: '語り始めた',
          arguments: [
            {
              caseName: 'ガ',
              argIndex: 8,
              argText: '主人は',
              score: 0.9,
              isZeroPronoun: false,
            },
          ],
        },
      ];

      // 通常時: severity: 'warning', isIntentionalIrregularity: false
      const normalDiags = SyntacticLinterRules.checkLongDistanceDependencies(sentence, 0, pasRelations);
      expect(normalDiags.length).toBe(1);
      expect(normalDiags[0].severity).toBe('warning');
      expect(normalDiags[0].isIntentionalIrregularity).toBe(false);

      // 条件A（複層的リズム文体: polyphony >= 70 && cadence >= 45）
      const intentionalTelemetry = {
        texture: {
          showing: 40,
          staccato: 30,
          polyphony: 75.0,
          cadence: 50.0,
          confession: 30,
          tension: 0.3,
        },
        dynamics: { crestFactor: 1.5, maxDependencyDistance: 12 },
      };

      const intentionalDiags = SyntacticLinterRules.checkLongDistanceDependencies(
        sentence,
        0,
        pasRelations,
        intentionalTelemetry
      );
      expect(intentionalDiags.length).toBe(1);
      expect(intentionalDiags[0].severity).toBe('info');
      expect(intentionalDiags[0].isIntentionalIrregularity).toBe(true);
      expect(intentionalDiags[0].message).toContain('【文体的破格（意図的演出）】');
    });

    it('should support lintSentence and lintDocument with per-sentence telemetry', () => {
      const sentences = ['彼が猫が魚が好きだと言った。'];
      const docs = SyntacticLinterRules.lintDocument(sentences, undefined, {
        texture: {
          showing: 40,
          staccato: 30,
          polyphony: 30,
          confession: 20,
          tension: 0.2,
          cadence: 35.0, // 緩和発動
        },
        dynamics: { crestFactor: 1.0 },
      });

      // 助詞重複警告が抑止されている
      expect(docs.filter((d) => d.source === 'narrative-nano-linter:particle-repetition').length).toBe(0);
    });
  });
});

