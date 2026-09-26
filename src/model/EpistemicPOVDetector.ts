/**
 * narrative-nano: EpistemicPOVDetector.ts
 *
 * Epistemic Point of View (POV) & Cognitive Fog Violation Detector for Literary Prose.
 * Implements Phase 1/Phase 3 of Narrative-Nano specification:
 * - Direct Internal State Expression of non-POV characters (他者内面描写の直接叙述検知)
 * - Rapid Head-Hopping across adjacent narrative sentences (シーン内視点乱移動検知)
 * - Cognitive Fog & Epistemic Boundary Violation (視点人物の不可知領域・認知境界違反)
 * - Automatic Literary Reformulation Suggestions (推量・外見描写への言い換え提案)
 */

export type POVMode =
  | 'first_person'             // 一人称視点 (語り手「私/僕/俺」のみ内面可)
  | 'third_person_limited'      // 三人称限定視点 (特定のPOVキャラクターのみ内面可)
  | 'third_person_omniscient'   // 三人称全知視点 (神の視点、急激なHead-Hoppingのみ警告)
  | 'third_person_objective';   // 三人称客観視点 (カメラアイ、全員内面不可)

export interface POVConfig {
  mode: POVMode;
  povCharacter?: string;        // 三人称限定時の視点人物名 (例: 'カフカ', 'エドガー')
  narratorPronouns?: string[];  // 一人称時の語り手代名詞 (デフォルト: ['私', '僕', '俺', 'わたし', '自分'])
  sensitivity?: 'strict' | 'normal' | 'lenient';
}

export interface POVSentenceInput {
  index: number;
  text: string;
  isDialogue: boolean;          // 会話文「...」は発話者の主観表現として除外または別枠
  subjectCandidate?: string;     // ガ格/ハ格の主体人物 (ZeroPronounResolverまたは明示構文から)
  startOffset?: number;
  endOffset?: number;
}

export interface POVViolation {
  sentenceIndex: number;
  text: string;
  charStart?: number;
  charEnd?: number;
  violationType: 'unauthorized_internal_state' | 'head_hopping' | 'objective_mode_leak';
  culpritEntity: string;
  triggerPhrase: string;
  confidence: number;           // 0.0 - 1.0
  severity: 'error' | 'warning' | 'info';
  message: string;
  suggestion: string;
}

export interface POVAnalysisReport {
  mode: POVMode;
  povCharacter: string;
  totalSentences: number;
  dialogueCount: number;
  violations: POVViolation[];
  headHopCount: number;
  povConsistencyScore: number;  // 0.0 (破綻) - 1.0 (完璧)
}

/**
 * 心理・内面描写述語辞書 (感情、思考、知覚、無意識の感情)
 */
const INTERNAL_STATE_PREDICATES: Array<{ pattern: RegExp; baseName: string }> = [
  { pattern: /(思った|思い至った|考えた|企んだ|目論んだ|思案した)/, baseName: '思考' },
  { pattern: /(感じた|実感した|直感した|胸を締め付けられた|戦慄した)/, baseName: '感情・感覚' },
  { pattern: /(悔やんだ|後悔した|恥じた|羨んだ|妬んだ|憎んだ|憤った|歓喜した|嘲笑った|蔑んだ)/, baseName: '情動' },
  { pattern: /(心の中で|胸の内で|胸中で|内心)/, baseName: '内的独白' },
  { pattern: /(恐れた|怯えた|恐怖した|怖気づいた|安堵した|ほっとした)/, baseName: '心理状態' },
  { pattern: /(思い描いた|夢見た|思いを巡らせた|決意した|覚悟を決めた)/, baseName: '意思決定' },
];

/**
 * 推量・外見描写マーカー (これらが後続していれば他者内面の直接叙述とはみなさない)
 */
const CONJECTURAL_MITIGATION_PATTERNS = [
  /ように見えた/,
  /ような気がした/,
  /ような表情/,
  /そうに見えた/,
  /そうだった/,
  /に違いない/,
  /のだろう/,
  /かもしれない/,
  /かのように/,
  /様子だった/,
  /かの如く/,
];

export class EpistemicPOVDetector {
  private config: Required<POVConfig>;

  constructor(config?: POVConfig) {
    this.config = {
      mode: config?.mode || 'third_person_limited',
      povCharacter: config?.povCharacter || '',
      narratorPronouns: config?.narratorPronouns || ['私', '僕', '俺', 'わたし', 'あたし', '自分', '我'],
      sensitivity: config?.sensitivity || 'normal',
    };
  }

  /**
   * 視点設定を動的に更新
   */
  public updateConfig(newConfig: Partial<POVConfig>): void {
    if (newConfig.mode) this.config.mode = newConfig.mode;
    if (newConfig.povCharacter !== undefined) this.config.povCharacter = newConfig.povCharacter;
    if (newConfig.narratorPronouns) this.config.narratorPronouns = newConfig.narratorPronouns;
    if (newConfig.sensitivity) this.config.sensitivity = newConfig.sensitivity;
  }

  /**
   * 文列を受け取り、POV違反およびHead-hoppingを検出
   */
  public analyzeSentences(sentences: POVSentenceInput[]): POVAnalysisReport {
    const violations: POVViolation[] = [];
    let headHopCount = 0;
    let dialogueCount = 0;

    let effectivePOVCharacter = this.config.povCharacter;
    if (!effectivePOVCharacter && this.config.mode === 'first_person') {
      effectivePOVCharacter = this.config.narratorPronouns[0] || '私';
    }

    let lastInternalStateCharacter: string | null = null;
    let lastInternalStateSentenceIndex = -1;

    for (const sent of sentences) {
      if (sent.isDialogue) {
        dialogueCount++;
        continue; // 会話文は台詞主の発話としてスキップ
      }

      const cleanText = sent.text.trim();
      if (!cleanText) continue;

      // 1. 内面描写述語の検出
      const internalMatch = this.detectInternalState(cleanText);
      if (!internalMatch) continue;

      // 推量・外見描写による緩和判定
      if (this.hasConjecturalMitigation(cleanText)) {
        continue;
      }

      // 主体の判定
      const subject = this.resolveSentenceSubject(sent, cleanText);

      // 2. 視点モード別ルール検証
      if (this.config.mode === 'third_person_objective') {
        // 客観視点では誰であっても内面描写は漏洩
        violations.push({
          sentenceIndex: sent.index,
          text: sent.text,
          violationType: 'objective_mode_leak',
          culpritEntity: subject || '人物',
          triggerPhrase: internalMatch.phrase,
          confidence: 0.95,
          severity: 'error',
          message: `三人称客観（カメラアイ）視点ですが、「${internalMatch.phrase}」という直接内面描写が含まれています。`,
          suggestion: this.generateSuggestion(internalMatch.phrase, subject || 'その人物'),
        });
        continue;
      }

      const isAllowedSubject = this.isSubjectAllowedPOV(subject, effectivePOVCharacter);

      if (!isAllowedSubject) {
        // 視点人物以外による直接内面描写 (POV Leakage)
        violations.push({
          sentenceIndex: sent.index,
          text: sent.text,
          violationType: 'unauthorized_internal_state',
          culpritEntity: subject || '視点外の人物',
          triggerPhrase: internalMatch.phrase,
          confidence: 0.9,
          severity: this.config.sensitivity === 'strict' ? 'error' : 'warning',
          message: this.config.mode === 'first_person'
            ? `一人称視点（語り手: ${effectivePOVCharacter}）ですが、他者（${subject || '視点外人物'}）の内面「${internalMatch.phrase}」が直接描写されています。`
            : `三人称限定視点（焦点: ${effectivePOVCharacter || '視点人物'}）ですが、別人物（${subject || '他者'}）の内面「${internalMatch.phrase}」が描写されています。`,
          suggestion: this.generateSuggestion(internalMatch.phrase, subject),
        });
      } else {
        // 許可された視点人物の内面描写の場合、Head-hoppingのチェック
        if (
          lastInternalStateCharacter &&
          lastInternalStateCharacter !== subject &&
          sent.index - lastInternalStateSentenceIndex <= 2
        ) {
          headHopCount++;
          if (this.config.mode !== 'third_person_omniscient' || this.config.sensitivity === 'strict') {
            violations.push({
              sentenceIndex: sent.index,
              text: sent.text,
              violationType: 'head_hopping',
              culpritEntity: subject || '',
              triggerPhrase: internalMatch.phrase,
              confidence: 0.8,
              severity: 'warning',
              message: `直前の文（${lastInternalStateCharacter}の内面）から急激に視点人物（${subject}）へ内面が切り替わっています（Head-Hopping）。`,
              suggestion: `段落を改めるか、視点人物の統一を検討してください。`,
            });
          }
        }

        lastInternalStateCharacter = subject;
        lastInternalStateSentenceIndex = sent.index;
      }
    }

    const narrativeCount = sentences.length - dialogueCount;
    const penalty = violations.reduce((acc, v) => acc + (v.severity === 'error' ? 0.25 : 0.1), 0);
    const povConsistencyScore = Math.max(0, Math.min(1.0, 1.0 - (narrativeCount > 0 ? penalty / Math.max(1, narrativeCount / 5) : 0)));

    return {
      mode: this.config.mode,
      povCharacter: effectivePOVCharacter,
      totalSentences: sentences.length,
      dialogueCount,
      violations,
      headHopCount,
      povConsistencyScore: Number(povConsistencyScore.toFixed(3)),
    };
  }

  private detectInternalState(text: string): { phrase: string; baseName: string } | null {
    for (const item of INTERNAL_STATE_PREDICATES) {
      const m = item.pattern.exec(text);
      if (m) {
        return { phrase: m[0], baseName: item.baseName };
      }
    }
    return null;
  }

  private hasConjecturalMitigation(text: string): boolean {
    return CONJECTURAL_MITIGATION_PATTERNS.some((pattern) => pattern.test(text));
  }

  private resolveSentenceSubject(sent: POVSentenceInput, text: string): string {
    if (sent.subjectCandidate) {
      return sent.subjectCandidate;
    }
    // 簡単な構文解析フォールバック (「〜は」「〜が」を抽出)
    const match = /(?:^|[、。])([一-龠々ぁ-んァ-ヶa-zA-Z0-9]+)[はが]/.exec(text);
    if (match && match[1]) {
      return match[1];
    }
    return '';
  }

  private isSubjectAllowedPOV(subject: string, effectivePOVCharacter: string): boolean {
    if (!subject) {
      // 主語省略時: 一人称ならデフォルトで語り手（許可）、限定視点で指定ありなら視点人物と推定
      return true;
    }

    if (this.config.mode === 'first_person') {
      return this.config.narratorPronouns.includes(subject) || subject === effectivePOVCharacter;
    }

    if (this.config.mode === 'third_person_limited') {
      if (!effectivePOVCharacter) return true; // 視点人物未定義なら許容
      return subject === effectivePOVCharacter;
    }

    if (this.config.mode === 'third_person_omniscient') {
      return true; // 全知視点は誰の内面でも構文上は許可
    }

    return false;
  }

  private generateSuggestion(triggerPhrase: string, subject?: string): string {
    const subjStr = subject ? `${subject}は` : '';
    if (triggerPhrase.includes('思った') || triggerPhrase.includes('考えた')) {
      return `${subjStr}〜と思っているように見えた / 〜といった様子を見せた`;
    }
    if (triggerPhrase.includes('感じた') || triggerPhrase.includes('恐怖した') || triggerPhrase.includes('恐れた')) {
      return `${subjStr}恐怖に顔を強張らせた / 〜そうに身を震わせた`;
    }
    if (triggerPhrase.includes('心の中で') || triggerPhrase.includes('胸の内で')) {
      return `直接の内的独白を避け、外見の仕草や表情描写に置き換えることを推奨します。`;
    }
    return `${subjStr}〜そうに見えた（外見・推量表現への言い換え）`;
  }
}
