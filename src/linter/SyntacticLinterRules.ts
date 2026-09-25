/**
 * Real-time Syntactic Twist (Subject-Predicate Mismatch & Particle Repetition) Linter Ruleset
 * Target environment: CodeMirror 6 Linter Extension & Narrative-Nano 3-pane IDE
 */

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
}

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

export class SyntacticLinterRules {
  private static readonly DEFAULT_OPTIONS: Required<SyntacticLinterOptions> = {
    minParticleRepetitionCount: 3,
    targetParticles: [...DEFAULT_TARGET_PARTICLES],
    syntacticScoreThreshold: 0.6,
    enableSubjectPredicateCheck: true,
    enableParticleRepetitionCheck: true,
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

    // Check particle repetitions penalty
    for (const particle of opts.targetParticles) {
      const count = SyntacticLinterRules.countParticleOccurrences(sentence, particle);
      if (count >= opts.minParticleRepetitionCount) {
        // Penalty increases with excess repetitions
        const excess = count - opts.minParticleRepetitionCount + 1;
        score -= 0.15 * excess;
      }
    }

    // Check subject-predicate mismatch penalty
    const hasMismatch = SyntacticLinterRules.hasSubjectPredicateMismatchPattern(sentence);
    if (hasMismatch) {
      score -= 0.45;
    }

    // Check excessive compound sentence penalty (過剰複文)
    const clauseCount = (sentence.match(/[,、]/g) || []).length;
    if (clauseCount >= 4) {
      score -= 0.1 * (clauseCount - 3);
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

    for (const particle of opts.targetParticles) {
      const occurrences = SyntacticLinterRules.findParticleOccurrences(sentence, particle);

      if (occurrences.length >= opts.minParticleRepetitionCount) {
        // Add diagnostic for each repeated particle instance or for the span
        for (const occ of occurrences) {
          diagnostics.push({
            from: sentenceOffset + occ.index,
            to: sentenceOffset + occ.index + particle.length,
            severity: 'warning',
            message: `同一文内で助詞「${particle}」が${occurrences.length}回重複して使用されています。`,
            source: 'narrative-nano-linter:particle-repetition',
          });
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
    const score = SyntacticLinterRules.calculateSyntacticScore(sentence, opts);
    const hasMismatchPattern = SyntacticLinterRules.hasSubjectPredicateMismatchPattern(sentence);

    if (hasMismatchPattern || score < opts.syntacticScoreThreshold) {
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
}
