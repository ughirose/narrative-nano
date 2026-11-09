/**
 * Real-time Syntactic Twist (Subject-Predicate Mismatch & Particle Repetition) Linter Ruleset
 * Target environment: CodeMirror 6 Linter Extension & Narrative-Nano 3-pane IDE
 */

import type { FusedProseTelemetry } from '@worldcraft/schema';
import type { PASPredicateRelation } from '../model/BiaffinePASHead.js';

export interface DiagnosticAction {
  name: string;
  apply: (view: any, from: number, to: number) => void;
}

export interface Diagnostic {
  from: number;
  to: number;
  severity: 'error' | 'warning' | 'info' | 'hint';
  message: string;
  source?: string;
  actions?: readonly DiagnosticAction[];
  isIntentionalIrregularity?: boolean;
}

export type SyntacticDiagnostic = Diagnostic;

export interface SyntacticLinterOptions {
  /**
   * Minimum particle repetition count within a single sentence to trigger a diagnostic.
   * Default: 3
   */
  minParticleRepetitionCount?: number;

  /**
   * List of target particles to inspect for repetition.
   * Default: ['が', 'を', 'に', 'で', 'の', 'と', 'から', 'より', 'へ', 'まで', 'は']
   */
  targetParticles?: string[];

  /**
   * Threshold for syntactic score below which a subject-predicate mismatch or syntactic distortion is flagged.
   * Range 0.0 to 1.0. Default: 0.6
   */
  syntacticScoreThreshold?: number;

  /**
   * Enable subject-predicate mismatch / twist detection.
   * Default: true
   */
  enableSubjectPredicateCheck?: boolean;

  /**
   * Enable particle repetition detection.
   * Default: true
   */
  enableParticleRepetitionCheck?: boolean;

  /**
   * Enable double negation detection.
   * Default: true
   */
  enableDoubleNegationCheck?: boolean;

  /**
   * Enable consecutive passive expression detection.
   * Default: true
   */
  enableConsecutivePassiveCheck?: boolean;

  /**
   * Kasane Stylistic Telemetry Block for dynamic relaxation and soft-landing.
   */
  kasaneTelemetry?: FusedProseTelemetry['kasane'];

  /**
   * Max dependency distance before flagging warning or intentional irregularity.
   * Default: 12
   */
  maxDependencyDistance?: number;
}

export interface SentenceAnalysisResult {
  sentence: string;
  from: number;
  to: number;
  syntacticScore: number;
  diagnostics: Diagnostic[];
}

export const DEFAULT_TARGET_PARTICLES = [
  'が',
  'を',
  'に',
  'で',
  'の',
  'と',
  'から',
  'より',
  'へ',
  'まで',
  'は',
] as const;

const NON_PASSIVE_WORDS = new Set(['これ', 'それ', 'あれ', 'だれ', 'あふれる', 'あふれた']);

export class SyntacticLinterRules {
  private static readonly DEFAULT_OPTIONS: Required<SyntacticLinterOptions> = {
    minParticleRepetitionCount: 3,
    targetParticles: [...DEFAULT_TARGET_PARTICLES],
    syntacticScoreThreshold: 0.6,
    enableSubjectPredicateCheck: true,
    enableParticleRepetitionCheck: true,
    enableDoubleNegationCheck: true,
    enableConsecutivePassiveCheck: true,
    kasaneTelemetry: undefined as any,
    maxDependencyDistance: 12,
  };

  /**
   * Main analysis entry point: analyzes full document text and returns CodeMirror 6 Diagnostics.
   */
  public static analyze(text: string, options?: SyntacticLinterOptions): Diagnostic[] {
    const opts: Required<SyntacticLinterOptions> = {
      ...SyntacticLinterRules.DEFAULT_OPTIONS,
      ...options,
    };

    const diagnostics: Diagnostic[] = [];
    const sentences = SyntacticLinterRules.splitSentences(text);

    for (const sent of sentences) {
      if (!sent.sentence.trim()) continue;

      // 1. Particle Repetition Check
      if (opts.enableParticleRepetitionCheck) {
        const particleDiags = SyntacticLinterRules.detectParticleRepetition(
          sent.sentence,
          sent.from,
          opts
        );
        diagnostics.push(...particleDiags);
      }

      // 2. Subject-Predicate Mismatch / Twist Check
      if (opts.enableSubjectPredicateCheck) {
        const spDiags = SyntacticLinterRules.detectSubjectPredicateMismatch(
          sent.sentence,
          sent.from,
          opts
        );
        diagnostics.push(...spDiags);
      }

      // 3. Double Negation Check
      if (opts.enableDoubleNegationCheck) {
        const dnDiags = SyntacticLinterRules.detectDoubleNegation(
          sent.sentence,
          sent.from
        );
        diagnostics.push(...dnDiags);
      }

      // 4. Consecutive Passive Check
      if (opts.enableConsecutivePassiveCheck) {
        const cpDiags = SyntacticLinterRules.detectConsecutivePassive(
          sent.sentence,
          sent.from
        );
        diagnostics.push(...cpDiags);
      }

      // 5. Long Distance Dependencies Check (§3.3 & §4.3)
      if (opts.kasaneTelemetry) {
        const longDepDiags = SyntacticLinterRules.checkLongDistanceDependencies(
          sent.sentence,
          sent.from,
          undefined,
          opts.kasaneTelemetry,
          opts.maxDependencyDistance
        );
        diagnostics.push(...longDepDiags);
      }
    }

    return diagnostics;
  }

  /**
   * Computes comprehensive syntactic quality score (0.0 to 1.0) for a single sentence.
   */
  public static calculateSyntacticScore(sentence: string, options?: SyntacticLinterOptions): number {
    const opts: Required<SyntacticLinterOptions> = {
      ...SyntacticLinterRules.DEFAULT_OPTIONS,
      ...options,
    };

    let score = 1.0;

    // Check particle repetitions penalty (evaluated per clause to respect compound sentences)
    const clauses = SyntacticLinterRules.splitClauses(sentence);
    for (const particle of opts.targetParticles) {
      let maxCount = 0;
      for (const cl of clauses) {
        const cOccs = SyntacticLinterRules.findParticleOccurrences(cl.clause, particle).length;
        if (cOccs > maxCount) maxCount = cOccs;
      }
      const minCount = particle === 'の' ? Math.max(opts.minParticleRepetitionCount, 4) : opts.minParticleRepetitionCount;
      if (maxCount >= minCount) {
        // Penalty increases with excess repetitions
        const excess = maxCount - minCount + 1;
        score -= 0.15 * excess;
      }
    }

    // Check subject-predicate mismatch penalty
    const hasMismatch = SyntacticLinterRules.hasSubjectPredicateMismatchPattern(sentence);
    if (hasMismatch) {
      score -= 0.45;
    }

    // Check double negation penalty
    const doubleNegations = SyntacticLinterRules.detectDoubleNegation(sentence, 0);
    if (doubleNegations.length > 0) {
      score -= 0.25 * doubleNegations.length;
    }

    // Check consecutive passive penalty
    const consecutivePassives = SyntacticLinterRules.detectConsecutivePassive(sentence, 0);
    if (consecutivePassives.length > 0) {
      score -= 0.2 * consecutivePassives.length;
    }

    // Check excessive compound sentence penalty (過剰複文: 極端な長文150字超かつ読点6個以上のみ)
    const clauseCount = (sentence.match(/[,、]/g) || []).length;
    if (sentence.length > 150 && clauseCount >= 6) {
      score -= 0.1 * (clauseCount - 5);
    }

    return Math.max(0.0, Math.min(1.0, score));
  }

  /**
   * Detects unnatural particle repetitions (>= 3 times) within a single sentence.
   */
  public static detectParticleRepetition(
    sentence: string,
    sentenceOffset: number,
    options?: SyntacticLinterOptions
  ): Diagnostic[] {
    const opts: Required<SyntacticLinterOptions> = {
      ...SyntacticLinterRules.DEFAULT_OPTIONS,
      ...options,
    };

    const diagnostics: Diagnostic[] = [];

    const cadence = opts.kasaneTelemetry?.texture?.cadence ?? opts.kasaneTelemetry?.dynamics?.cadence ?? 0;
    const isClassicalCadence = cadence >= 30.0;
    const baseMinCount = opts.minParticleRepetitionCount;
    const effectiveMinCount = isClassicalCadence ? Math.ceil(baseMinCount * 1.5) : baseMinCount;
    const noThreshold = isClassicalCadence ? Math.ceil(3 * 1.5) : 3;

    for (const particle of opts.targetParticles) {
      if (particle === 'の') {
        // Evaluate per clause/subclause to prevent false positives on multiple distinct nominal modifiers
        const clauseRegex = /[^、――──「」『』（）…：；\n]+(?:[、――──「」『』（）…：；\n]|$)/g;
        let cMatch: RegExpExecArray | null;
        while ((cMatch = clauseRegex.exec(sentence)) !== null) {
          const clauseText = cMatch[0];
          const clauseStart = cMatch.index;
          const occs = SyntacticLinterRules.findParticleOccurrences(clauseText, 'の');
          if (occs.length >= noThreshold) {
            let consecutiveChain = [occs[0]];
            for (let i = 1; i < occs.length; i++) {
              const prev = occs[i - 1].index;
              const curr = occs[i].index;
              const span = clauseText.slice(prev + 1, curr);
              if (/(?:[\u4E00-\u9FFF\u30A0-\u30FF\u3040-\u309F]+?)(?:が|を|は|に|で|へ|と|から|より|も)/.test(span) || /[\s\(\)「」]/.test(span)) {
                if (consecutiveChain.length >= noThreshold) {
                  const severity = consecutiveChain.length >= 4 ? 'warning' : 'info';
                  for (const occ of consecutiveChain) {
                    diagnostics.push({
                      from: sentenceOffset + clauseStart + occ.index,
                      to: sentenceOffset + clauseStart + occ.index + 1,
                      severity,
                      message: `同一名詞句内で助詞「の」が${consecutiveChain.length}回重複して使用されています。修飾関係を見直してください。`,
                      source: 'narrative-nano-linter:particle-repetition',
                    });
                  }
                }
                consecutiveChain = [occs[i]];
              } else {
                consecutiveChain.push(occs[i]);
              }
            }
            if (consecutiveChain.length >= noThreshold) {
              const severity = consecutiveChain.length >= 4 ? 'warning' : 'info';
              for (const occ of consecutiveChain) {
                diagnostics.push({
                  from: sentenceOffset + clauseStart + occ.index,
                  to: sentenceOffset + clauseStart + occ.index + 1,
                  severity,
                  message: `同一名詞句内で助詞「の」が${consecutiveChain.length}回重複して使用されています。修飾関係を見直してください。`,
                  source: 'narrative-nano-linter:particle-repetition',
                });
              }
            }
          }
        }
        continue;
      }

      const clauses = SyntacticLinterRules.splitClauses(sentence);
      for (const clauseItem of clauses) {
        const occurrences = SyntacticLinterRules.findParticleOccurrences(clauseItem.clause, particle);

        if (occurrences.length >= effectiveMinCount) {
          for (const occ of occurrences) {
            diagnostics.push({
              from: sentenceOffset + clauseItem.start + occ.index,
              to: sentenceOffset + clauseItem.start + occ.index + particle.length,
              severity: 'warning',
              message: `同一文内で助詞「${particle}」が${occurrences.length}回重複して使用されています。読者に単調・稚拙な印象を与える可能性があるため、助詞の省略や言い換えを検討してください。`,
              source: 'narrative-nano-linter:particle-repetition',
            });
          }
        }
      }
    }

    return diagnostics;
  }

  /**
   * Detects subject-predicate dependency mismatch or twist in a sentence.
   */
  public static detectSubjectPredicateMismatch(
    sentence: string,
    sentenceOffset: number,
    options?: SyntacticLinterOptions
  ): Diagnostic[] {
    const opts: Required<SyntacticLinterOptions> = {
      ...SyntacticLinterRules.DEFAULT_OPTIONS,
      ...options,
    };

    const diagnostics: Diagnostic[] = [];

    // Literary Dash Apposition Guard: If sentence uses dashes (――/──) as an appositive/parenthetical clause,
    // e.g. 「忘却の砦――かつて神話の時代に……絶壁の城塞である。」,
    // strip the parenthetical insertion to inspect the core sentence structure.
    let inspectionText = sentence;
    if (/(?:――|──)/.test(sentence)) {
      // If ends with copula (〜である, 〜だ) and subject is fronted with dash apposition, this is valid literary style
      if (/(?:――|──)[^――──]+(?:である|だ|だった|でした)[。！？\n]?$/.test(sentence)) {
        return [];
      }
      inspectionText = sentence.replace(/(?:――|──)[^――──]+(?:――|──)/g, '、');
    }

    const hasMismatchPattern = SyntacticLinterRules.hasSubjectPredicateMismatchPattern(inspectionText);

    if (hasMismatchPattern) {
      const score = SyntacticLinterRules.calculateSyntacticScore(sentence, opts);
      // Locate the subject/topic marker and sentence ending predicate if possible
      const topicMatch = sentence.match(/(?:[^\s。、]{1,20})(?:は|が|のは|としては)/);
      const from = sentenceOffset + (topicMatch?.index ?? 0);
      const to = sentenceOffset + sentence.length;

      diagnostics.push({
        from,
        to,
        severity: 'warning',
        message: `主語と述語の文末係り受けのねじれ（主述不整合）を検知しました。(構文スコア: ${(score * 100).toFixed(0)}/100)`,
        source: 'narrative-nano-linter:subject-predicate-mismatch',
      });
    }

    return diagnostics;
  }

  /**
   * Pattern analysis for subject-predicate mismatch (e.g., 「〜は、……と思ったからです。」)
   */
  private static hasSubjectPredicateMismatchPattern(sentence: string): boolean {
    const trimmed = sentence.trim();

    // Guard: Exclude valid literary dash apposition/definition sentences
    if (/(?:――|──)/.test(trimmed)) {
      return false;
    }

    // Pattern 1: Topic marker "〜は" or "〜が" combined with sentence-ending reason/cause copula "〜からです" / "〜ためです" / "〜だからです"
    // e.g., 「〜は、……と思ったからです。」 or 「彼の夢は、……優勝したからです。」
    const topicAndReasonMismatch = /^(?:(?![^。、]*[の理由原因目的]は).)*?[はが]\s*[,、].*?(?:と思ったからです|と考えたからです|と感じたからです|からです|ためです|だからです|からである|ためである)[。!！?？]?$/;

    // Pattern 2: Over-complex compound sentence twist ("〜は、……と思ったからです")
    const overComplexTwist = /[はが][,、].*?(?:と思った|と考え|と感じ).*?(?:からです|ためです)/;

    // Pattern 3: Direct subject noun ("夢は", "彼は", "目標は") with reason ending when noun is not reason/cause
    const nonReasonSubjectWithReasonEnding = /(?:[彼我君彼女私自分][はが]|夢[はが]|目標[はが]|計画[はが]|行動[はが]|思い[はが]).*?(?:からです|ためです)[。!！?？]?$/;

    return (
      topicAndReasonMismatch.test(trimmed) ||
      overComplexTwist.test(trimmed) ||
      nonReasonSubjectWithReasonEnding.test(trimmed)
    );
  }

  /**
   * Detects double negation patterns within a sentence.
   */
  public static detectDoubleNegation(
    sentence: string,
    sentenceOffset: number
  ): Diagnostic[] {
    const diagnostics: Diagnostic[] = [];
    const doubleNegationRegex =
      /(?:ないわけではない|ないわけじゃない|なくもない|なくはない|ないとは言えない|ないとは限らない|ざるを得ない|ないことはない|ないこともない|否定できない|否定はできない|ずにはいられない|ないではいられない|思わないでもない)/g;

    let match: RegExpExecArray | null;
    while ((match = doubleNegationRegex.exec(sentence)) !== null) {
      diagnostics.push({
        from: sentenceOffset + match.index,
        to: sentenceOffset + match.index + match[0].length,
        severity: 'warning',
        message: `二重否定「${match[0]}」が検出されました。簡潔で明瞭な肯定表現への見直しを推奨します。`,
        source: 'narrative-nano-linter:double-negation',
      });
    }

    return diagnostics;
  }

  /**
   * Detects consecutive passive expressions (2 or more) in a single sentence.
   */
  public static detectConsecutivePassive(
    sentence: string,
    sentenceOffset: number
  ): Diagnostic[] {
    const diagnostics: Diagnostic[] = [];
    const passiveRegex =
      /(?:[ぁ-んァ-ヶー一-龠]{1,8}?(?:(?:[あかさたなはまやらわがざだばぱ])(?:れる|れた|れて|れば)|(?:される|された|されて|せられる|せられた)|(?:られる|られた|られて|られれば)))/g;

    const occurrences: Array<{ index: number; length: number; text: string }> = [];
    let match: RegExpExecArray | null;
    while ((match = passiveRegex.exec(sentence)) !== null) {
      const text = match[0];
      // Filter out non-passive common words (e.g., あれこれ, それ, あふれる)
      if (NON_PASSIVE_WORDS.has(text)) {
        continue;
      }

      // Filter out hearsay/spontaneous predicate: 〜と伝えられる, 〜と言われる, etc.
      const surrounding = sentence.slice(Math.max(0, match.index - 5), match.index + text.length + 5);
      if (/と(?:伝えられ|言われ|考えられ|思われ|見られ)/.test(surrounding)) {
        continue;
      }
      occurrences.push({
        index: match.index,
        length: match[0].length,
        text,
      });
    }

    if (occurrences.length >= 2) {
      for (const occ of occurrences) {
        diagnostics.push({
          from: sentenceOffset + occ.index,
          to: sentenceOffset + occ.index + occ.length,
          severity: 'warning',
          message: `同一文内で受身表現（計${occurrences.length}箇所）が連続しています。動作主が曖昧になりやすいため能動態への変更を検討してください。`,
          source: 'narrative-nano-linter:consecutive-passive',
        });
      }
    }

    return diagnostics;
  }

  /**
   * Helper to split text into sentences while tracking document character offsets.
   */
  public static splitSentences(text: string): Array<{ sentence: string; from: number; to: number }> {
    const results: Array<{ sentence: string; from: number; to: number }> = [];
    const sentenceDelimiterRegex = /[。\n!！?？]/g;

    let start = 0;
    let match: RegExpExecArray | null;

    while ((match = sentenceDelimiterRegex.exec(text)) !== null) {
      const end = match.index + match[0].length;
      const sentence = text.slice(start, end);
      if (sentence.trim().length > 0) {
        results.push({
          sentence,
          from: start,
          to: end,
        });
      }
      start = end;
    }

    if (start < text.length) {
      const sentence = text.slice(start);
      if (sentence.trim().length > 0) {
        results.push({
          sentence,
          from: start,
          to: text.length,
        });
      }
    }

    return results;
  }

  /**
   * Helper to split a sentence into constituent clauses by punctuation (、, ――, ──, …, etc.)
   * and conjunctive particle boundaries to respect natural compound/complex sentence structures.
   */
  public static splitClauses(sentence: string): Array<{ clause: string; start: number; end: number }> {
    const results: Array<{ clause: string; start: number; end: number }> = [];
    const clauseRegex = /[^、――──「」『』（）…：；\n]+(?:[、――──「」『』（）…：；\n]|$)/g;
    let match: RegExpExecArray | null;

    while ((match = clauseRegex.exec(sentence)) !== null) {
      if (match[0].trim().length > 0) {
        results.push({
          clause: match[0],
          start: match.index,
          end: match.index + match[0].length,
        });
      }
    }

    if (results.length === 0 && sentence.trim().length > 0) {
      results.push({
        clause: sentence,
        start: 0,
        end: sentence.length,
      });
    }

    return results;
  }

  /**
   * Finds occurrences of a particle in a sentence with boundary awareness.
   */
  private static findParticleOccurrences(
    sentence: string,
    particle: string
  ): Array<{ index: number }> {
    const occurrences: Array<{ index: number }> = [];
    let searchPos = 0;

    while (searchPos < sentence.length) {
      const idx = sentence.indexOf(particle, searchPos);
      if (idx === -1) break;

      // Ensure boundary validation so multi-character non-particles aren't false-positived
      if (SyntacticLinterRules.isValidParticleOccurrence(sentence, idx, particle)) {
        occurrences.push({ index: idx });
      }

      searchPos = idx + particle.length;
    }

    return occurrences;
  }

  /**
   * Counts occurrences of a particle in a sentence.
   */
  private static countParticleOccurrences(sentence: string, particle: string): number {
    return SyntacticLinterRules.findParticleOccurrences(sentence, particle).length;
  }

  /**
   * Particle boundary check to prevent false positives in kanji or compound words.
   */
  private static isValidParticleOccurrence(
    sentence: string,
    index: number,
    particle: string
  ): boolean {
    const prevChar = index > 0 ? sentence[index - 1] : '';
    const nextChar = index + particle.length < sentence.length ? sentence[index + particle.length] : '';

    // Check if particle is 'が' inside words like 'および', 'だが', 'それが'
    // particles normally attach to nouns/verbs/adjectives or punctuation
    if (particle === 'が') {
      if (sentence.slice(Math.max(0, index - 2), index + 1) === 'および') return false;
    }

    // Avoid false positive when particle is inside double quotes or specific non-particle prefix/suffix
    if (particle === 'の') {
      // Exclude formal nouns, compound particles, and adjectives starting with 'の': のみ, ので, のに, のは, のが, のを, のも, のだ, のか, のどけき, のどか
      const rest = sentence.slice(index);
      if (/^の(?:み|で|に|は|が|を|も|だ|か|よ|ね|どけ|どか)/.test(rest)) {
        return false;
      }
      // Exclude lexicalized fixed compound nouns containing 'の': 手のひら, 目の前, 女の子, 男の子
      if (index >= 1 && /^(?:手のひら|目の前|女の子|男の子)/.test(sentence.slice(index - 1))) {
        return false;
      }
      // Exclude words ending in 'の' like 'ものの', 'この', 'その', 'あの', 'どの', 'ほんの', 'かの'
      const prevTwoChars = sentence.slice(Math.max(0, index - 1), index + 1);
      if (['この', 'その', 'あの', 'どの', 'もの', 'かの'].includes(prevTwoChars)) {
        return false;
      }
      const prevThreeChars = sentence.slice(Math.max(0, index - 2), index + 1);
      if (['ものの', 'ほんの'].includes(prevThreeChars)) {
        return false;
      }
    }

    if (particle === 'と') {
      // 1. Exclude quoted particle literals: 「と」, 『と』, "と", 'と'
      if (
        (index > 0 && ['「', '『', '“', '"', '‘', '`'].includes(prevChar)) ||
        (index + 1 < sentence.length && ['」', '』', '”', '"', '’', '`'].includes(nextChar))
      ) {
        return false;
      }

      // 2. Exclude nouns starting with 'と' (formal nouns, time, place):
      // とき (時), ところ (所), とおり (通り), となり (隣), とも (友/共), とちゅう (途中)
      const rest = sentence.slice(index);
      if (/^と(?:き|ころ|おり|なり|ちゅう)/.test(rest)) {
        return false;
      }

      // 3. Exclude adverbs ending in 'と':
      // ちゃんと, ひょっとすると, きっと, ふと, そっと, じっと, やっと, もっと, ずっと, 堂々と, 忽然と, 凛と, パッと, etc.
      const before12 = sentence.slice(Math.max(0, index - 10), index + particle.length);
      if (
        /ひょっとする?と$/.test(before12) ||
        /(?:きっ|ふ|ふっ|そっ|じっ|やっ|もっ|ずっ|堂々|凛|忽然|パッ|スッ|ハッ|バッ|サッ|じっくり|はっきり|くっきり|ゆったり|しっかりと?|ちゃん|なんと|まったく)と$/.test(before12)
      ) {
        return false;
      }

      // 4. Exclude compound particles and auxiliaries:
      // として, としては, とともに, にとって, に対して, といった, という, のこと, とする, とした, とみる, とみられる
      if (/^と(?:して|ともに|いう|いった|のこと|する|した|みる|みられる)/.test(rest)) {
        return false;
      }

      // 5. Exclude quote/thought verbs: 〜と思う, 〜と考え, 〜と言う, 〜と聞く, 〜と感じる
      if (/^と(?:は思|思|は考|考|は言|言|は聞|聞|は感|感)/.test(rest)) {
        return false;
      }

      // 6. Exclude 'ことと' inside 'こととして'
      const prevTwo = sentence.slice(Math.max(0, index - 2), index);
      if (prevTwo === 'こと' && /^として/.test(rest)) {
        return false;
      }

      // 7. Exclude parallel/listing particles: とか (〜とか〜とか)
      if (/^とか/.test(rest)) {
        return false;
      }
    }

    return true;
  }

  /**
   * CodeMirror 6 Linter Extension factory function.
   */
  public static createLinterExtension(options?: SyntacticLinterOptions) {
    return (view: { state: { doc: { toString(): string } } } | any): Diagnostic[] => {
      const docText =
        typeof view?.state?.doc?.toString === 'function'
          ? view.state.doc.toString()
          : typeof view === 'string'
          ? view
          : '';

      return SyntacticLinterRules.analyze(docText, options);
    };
  }

  /**
   * Checks long-distance dependencies between subject/modifier and predicate (§3.3 & §4.3).
   * Flags extreme distances (> maxDependencyDistance), but soft-lands to intentional irregularity (破格)
   * under high polyphony & cadence, high tension & confession, or high crestFactor.
   */
  public static checkLongDistanceDependencies(
    sentence: string,
    sentenceOffset: number,
    pasRelations?: PASPredicateRelation[],
    kasaneTelemetry?: FusedProseTelemetry['kasane'],
    maxDependencyDistance: number = 12
  ): Diagnostic[] {
    const diagnostics: Diagnostic[] = [];
    const maxDepDist = kasaneTelemetry?.dynamics?.maxDependencyDistance ?? maxDependencyDistance;

    const texture = kasaneTelemetry?.texture;
    const dynamics = kasaneTelemetry?.dynamics;
    const polyphony = texture?.polyphony ?? 0;
    const cadence = texture?.cadence ?? dynamics?.cadence ?? 0;
    const confession = texture?.confession ?? 0;
    const rawTension = texture?.tension ?? 0;
    const tension = rawTension > 1.0 ? rawTension / 100.0 : rawTension;
    const crestFactor = dynamics?.crestFactor ?? 1.0;

    // Condition A: Complex rhythmic style (polyphony >= 70.0 && cadence >= 45.0)
    // Condition B: High tension/emotional climax (tension >= 0.7 && confession >= 70.0) or (crestFactor >= 2.5)
    const isIntentional =
      (polyphony >= 70.0 && cadence >= 45.0) ||
      (tension >= 0.7 && confession >= 70.0) ||
      (crestFactor >= 2.5);

    // If pasRelations provided, check distance between predicate and arguments
    if (pasRelations && pasRelations.length > 0) {
      for (const rel of pasRelations) {
        for (const arg of rel.arguments) {
          if (!arg.isZeroPronoun && arg.argIndex !== null) {
            const dist = Math.abs(rel.predicateIndex - arg.argIndex);
            if (dist > maxDepDist) {
              const subjectText = arg.argText ?? sentence.slice(arg.argIndex, Math.min(sentence.length, arg.argIndex + 4));
              const predText = rel.predicateText ?? sentence.slice(rel.predicateIndex, Math.min(sentence.length, rel.predicateIndex + 4));

              if (isIntentional) {
                diagnostics.push({
                  from: sentenceOffset + Math.min(arg.argIndex, rel.predicateIndex),
                  to: sentenceOffset + Math.max(arg.argIndex, rel.predicateIndex) + 1,
                  severity: 'info',
                  isIntentionalIrregularity: true,
                  message: `【文体的破格（意図的演出）】主語・修飾語「${subjectText}」と述語「${predText}」の距離が離れていますが、高い文体張力・律動を伴う意図的な破格表現と判定されました。`,
                  source: 'narrative-nano-linter:long-dependency-distance',
                });
              } else {
                diagnostics.push({
                  from: sentenceOffset + Math.min(arg.argIndex, rel.predicateIndex),
                  to: sentenceOffset + Math.max(arg.argIndex, rel.predicateIndex) + 1,
                  severity: 'warning',
                  isIntentionalIrregularity: false,
                  message: `主語・修飾語「${subjectText}」と述語「${predText}」の係り受け距離（${dist}文字）が長すぎます。主述のねじれや読解困難を招く可能性があります。`,
                  source: 'narrative-nano-linter:long-dependency-distance',
                });
              }
            }
          }
        }
      }
    } else {
      // Fallback heuristics: find topic marker (〜は, 〜が) and sentence ending predicate
      const topicMatch = sentence.match(/(?:^|[、。])([一-龠々ぁ-んァ-ヶa-zA-Z0-9]+[はが])/);
      if (topicMatch && topicMatch.index !== undefined) {
        const topicPos = topicMatch.index + (topicMatch[0].startsWith('、') || topicMatch[0].startsWith('。') ? 1 : 0);
        const endPos = sentence.length - 1;
        const dist = endPos - topicPos;

        if (dist > maxDepDist + 8) {
          const subjectText = topicMatch[1];
          const predText = sentence.slice(Math.max(0, sentence.length - 5), sentence.length);

          if (isIntentional) {
            diagnostics.push({
              from: sentenceOffset + topicPos,
              to: sentenceOffset + endPos,
              severity: 'info',
              isIntentionalIrregularity: true,
              message: `【文体的破格（意図的演出）】主語・修飾語「${subjectText}」と述語「${predText}」の距離が離れていますが、高い文体張力・律動を伴う意図的な破格表現と判定されました。`,
              source: 'narrative-nano-linter:long-dependency-distance',
            });
          } else {
            diagnostics.push({
              from: sentenceOffset + topicPos,
              to: sentenceOffset + endPos,
              severity: 'warning',
              isIntentionalIrregularity: false,
              message: `主語・修飾語「${subjectText}」と述語「${predText}」の係り受け距離（${dist}文字）が長すぎます。主述のねじれや読解困難を招く可能性があります。`,
              source: 'narrative-nano-linter:long-dependency-distance',
            });
          }
        }
      }
    }

    return diagnostics;
  }

  /**
   * Lint a single sentence with optional PAS relations and Kasane telemetry.
   */
  public static lintSentence(
    sentence: string,
    index: number = 0,
    pasRelations?: PASPredicateRelation[],
    kasaneTelemetry?: FusedProseTelemetry['kasane']
  ): Diagnostic[] {
    const diags = SyntacticLinterRules.analyze(sentence, {
      kasaneTelemetry,
    });

    if (pasRelations && pasRelations.length > 0) {
      // Remove any heuristic fallback long-dependency-distance diagnostic from analyze() when explicit PAS relations are provided
      const filteredDiags = diags.filter(
        (d) => d.source !== 'narrative-nano-linter:long-dependency-distance'
      );
      const longDepDiags = SyntacticLinterRules.checkLongDistanceDependencies(
        sentence,
        0,
        pasRelations,
        kasaneTelemetry
      );
      filteredDiags.push(...longDepDiags);
      return filteredDiags;
    }

    return diags;
  }

  /**
   * Lint full document sentences with per-sentence PAS relations and Kasane telemetry.
   */
  public static lintDocument(
    sentences: string[],
    pasRelationsPerSentence?: (PASPredicateRelation[] | undefined)[],
    kasaneTelemetryPerSentence?: (FusedProseTelemetry['kasane'] | undefined)[] | FusedProseTelemetry['kasane']
  ): Diagnostic[] {
    const diagnostics: Diagnostic[] = [];
    let docOffset = 0;

    for (let i = 0; i < sentences.length; i++) {
      const sent = sentences[i];
      const pas = pasRelationsPerSentence?.[i];
      const telemetry = Array.isArray(kasaneTelemetryPerSentence)
        ? kasaneTelemetryPerSentence[i]
        : kasaneTelemetryPerSentence;

      const sentDiags = SyntacticLinterRules.lintSentence(sent, i, pas, telemetry);
      for (const d of sentDiags) {
        diagnostics.push({
          ...d,
          from: d.from + docOffset,
          to: d.to + docOffset,
        });
      }

      docOffset += sent.length;
    }

    return diagnostics;
  }
}
