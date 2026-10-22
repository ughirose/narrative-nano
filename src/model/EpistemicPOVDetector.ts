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

import type { FusedProseTelemetry, POVViolationType } from '@worldcraft/schema';

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
  kasaneTelemetry?: FusedProseTelemetry['kasane'];
}

export interface POVViolation {
  sentenceIndex: number;
  text: string;
  charStart?: number;
  charEnd?: number;
  violationType: POVViolationType | string;
  culpritEntity: string;
  triggerPhrase: string;
  confidence: number;           // 0.0 - 1.0
  severity: 'error' | 'warning' | 'info' | string;
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
  { pattern: /(思った|思って|思い至った|考えた|考えて|企んだ|企んで|目論んだ|目論んで|思案した|思案して|確信した|確信して|疑った|疑って)/, baseName: '思考' },
  { pattern: /(感じた|感じて|実感した|実感して|直感した|直感して|胸を締め付けられた|戦慄した|戦慄して)/, baseName: '感情・感覚' },
  { pattern: /(悔やんだ|悔やんで|後悔した|後悔して|恥じた|恥じて|羨んだ|羨んで|妬んだ|妬んで|憎んだ|憎んで|憤った|憤って|歓喜した|歓喜して|嘲笑った|嘲笑って|蔑んだ|蔑んで|軽蔑し|軽蔑して|恨んだ|恨んで|嫌悪した|嫌悪して)/, baseName: '情動' },
  { pattern: /(心の中で|胸の内で|胸中で|内心)/, baseName: '内的独白' },
  { pattern: /(恐れた|恐れて|怯えた|怯えて|恐怖した|恐怖して|怖気づいた|安堵した|安堵して|ほっとした)/, baseName: '心理状態' },
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
  /ようだった/,
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
   * Kasane文体テレメトリ（文単位またはグローバル）を受け取り、動的ベイズ補正およびFID・認知漏洩判定を強化
   */
  public analyzeSentences(
    sentences: POVSentenceInput[],
    globalTelemetry?: FusedProseTelemetry['kasane']
  ): POVAnalysisReport {
    const violations: POVViolation[] = [];
    let headHopCount = 0;
    let dialogueCount = 0;

    let effectivePOVCharacter = this.config.povCharacter;
    if (!effectivePOVCharacter && this.config.mode === 'first_person') {
      effectivePOVCharacter = this.config.narratorPronouns[0] || '私';
    }

    let lastInternalStateCharacter: string | null = null;
    let lastInternalStateSentenceIndex = -1;

    for (let i = 0; i < sentences.length; i++) {
      const sent = sentences[i];
      if (sent.isDialogue) {
        dialogueCount++;
        continue; // 会話文は台詞主の発話としてスキップ
      }

      const cleanText = sent.text.trim();
      if (!cleanText) continue;

      // 該当文またはグローバルのKasaneテレメトリを取得
      const telemetry = sent.kasaneTelemetry || globalTelemetry;
      const texture = telemetry?.texture;
      const showing = texture?.showing ?? 50.0;
      const speculation = texture?.speculation ?? 0.0;
      const staccato = texture?.staccato ?? 50.0;
      const confession = texture?.confession ?? 0.0;
      const rawTension = texture?.tension ?? 0.0;
      const tension = rawTension > 1.0 ? Math.min(1.0, rawTension / 100.0) : rawTension;

      // テンション変調判定
      const prevTelemetry = i > 0 ? (sentences[i - 1].kasaneTelemetry || globalTelemetry) : undefined;
      const rawPrevTension = prevTelemetry?.texture.tension;
      const prevTension =
        rawPrevTension !== undefined
          ? rawPrevTension > 1.0
            ? Math.min(1.0, rawPrevTension / 100.0)
            : rawPrevTension
          : undefined;
      const deltaTension = prevTension !== undefined ? Math.abs(tension - prevTension) : 0.0;
      const hasTensionModulation = (deltaTension > 0.3) || (tension >= 0.3);

      // 主体の判定
      const subject = this.resolveSentenceSubject(sent, cleanText);

      // 機能 A: 自由間接話法（FID）の軟着陸 (§3.1 & §4.1 過去回想ガード)
      // third_person_limited において、confession >= 65.0 (過去回想時は >= 80.0) かつ tension 変調があり、
      // 疑問・反語や内面表出が検出された場合（明白な別人物主語でない場合）、severity: 'info' として軟着陸
      const isPastRecollection = /(?:十年前|あの日|当時は|あの頃|その頃|かつて)[^。]*?(?:だった|であった|いた|た。)/.test(cleanText);
      const fidConfessionThreshold = isPastRecollection ? 80.0 : 65.0;
      const isFIDMarker = this.hasFIDMarkers(cleanText);
      const isOtherSubjectExplicit = Boolean(subject && effectivePOVCharacter && subject !== effectivePOVCharacter);

      if (
        this.config.mode === 'third_person_limited' &&
        confession >= fidConfessionThreshold &&
        hasTensionModulation &&
        isFIDMarker &&
        !isOtherSubjectExplicit
      ) {
        const internalMatch = this.detectInternalState(cleanText);
        violations.push({
          sentenceIndex: sent.index,
          text: sent.text,
          violationType: 'free_indirect_discourse',
          culpritEntity: effectivePOVCharacter || subject || '視点人物',
          triggerPhrase: internalMatch?.phrase || cleanText,
          confidence: 0.85,
          severity: 'info',
          message: `自由間接話法（FID: 内面浸透）が検出されました。地の文に視点人物の感情・思考が自然に溶け込んでいます。`,
          suggestion: `文体演出として受容されます（重大度: info）。`,
        });
        lastInternalStateCharacter = effectivePOVCharacter || subject;
        lastInternalStateSentenceIndex = sent.index;
        continue;
      }

      // 1. 内面描写述語の検出
      const internalMatch = this.detectInternalState(cleanText);
      if (!internalMatch) continue;

      // 5文スライディングウィンドウによる局所微分スパイク判定 (§4.2)
      let hasSpeculationSpike = false;
      if (i >= 1) {
        const windowStart = Math.max(0, i - 4);
        const windowSentences = sentences.slice(windowStart, i);
        const prevSpecList = windowSentences
          .map((s) => (s.kasaneTelemetry || globalTelemetry)?.texture?.speculation)
          .filter((v): v is number => v !== undefined);
        if (prevSpecList.length > 0) {
          const minPrev = Math.min(...prevSpecList);
          if (speculation - minPrev >= 25.0 && speculation >= 60.0) {
            hasSpeculationSpike = true;
          }
        }
      }

      // 機能 B: Showing 欠如 ＋ Speculation 過多判定 (§3.2 & §4.2)
      const isExcessiveSpeculation =
        (showing <= 20.0 && speculation >= 60.0 && staccato <= 25.0) || hasSpeculationSpike;

      // 推量・外見描写による緩和判定
      const hasMitigation = this.hasConjecturalMitigation(cleanText);

      const isAllowedSubject = this.isSubjectAllowedPOV(subject, effectivePOVCharacter);

      if (hasMitigation) {
        if (isExcessiveSpeculation) {
          // 推量免除ペナルティ適用: 免除を無効化し、認知漏洩として検知
          if (!isAllowedSubject || this.config.mode === 'third_person_objective') {
            const baseConfidence = 0.85;
            const beta = 0.5;
            const epsilon = 1.0;
            const scoreLeak = Math.min(
              1.0,
              baseConfidence * (1.0 + beta * (speculation / (showing + epsilon)))
            );

            violations.push({
              sentenceIndex: sent.index,
              text: sent.text,
              violationType: 'epistemic_leak_excessive_speculation',
              culpritEntity: subject || '視点外の人物',
              triggerPhrase: internalMatch.phrase,
              confidence: Number(scoreLeak.toFixed(3)),
              severity: 'warning',
              message: `【説明過多・認知漏洩の兆候】外見描写（Showing: ${showing.toFixed(1)}）が不足したまま推量（Speculation: ${speculation.toFixed(1)}）による内面説明が連続しています。登場人物の客観的挙動や身体反応の描写への置き換えを推奨します。`,
              suggestion: `登場人物の客観的挙動や身体反応の描写への置き換えを推奨します。`,
            });
            continue;
          }
        } else {
          // 通常の推量免除
          continue;
        }
      }

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
    const penalty = violations.reduce((acc, v) => {
      if (v.severity === 'error') return acc + 0.25;
      if (v.severity === 'warning') return acc + 0.1;
      return acc; // info は減点なし
    }, 0);
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

  private hasFIDMarkers(text: string): boolean {
    const fidPatterns = [
      /なぜ.*(?:か|のか|だろう|ねばならない)/,
      /どうして.*(?:か|のか|だろう)/,
      /.*だろうか/,
      /.*ねばならないのか/,
      /なんと.*(?:だろう|ことだ|ことか|のだろう)/,
      /まさか.*/,
      /.*に違いない/,
      /.*はずがない/,
      /.*ではないか/,
      /.*てたまるか/,
      /.*ものか/,
      /.*どうすれば.*(?:か|のか|だろう|のだろう)/,
      /[？?]/,
    ];
    return fidPatterns.some((pattern) => pattern.test(text));
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
