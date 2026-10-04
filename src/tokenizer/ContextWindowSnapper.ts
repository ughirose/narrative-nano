/**
 * @file ContextWindowSnapper.ts
 *
 * Sliding Window Extractor with Sentence Boundary & Bracket Consistency Snap.
 * Extracts context slices (typically 128~256 characters) around an arbitrary
 * caret position while automatically adjusting boundaries to prevent sentence
 * fragmentation, unclosed brackets, and Japanese IME composition noise.
 */

export interface ContextWindowSnapperOptions {
  /** Minimum window size in characters (default: 128) */
  minWindowSize?: number;
  /** Maximum window size in characters (default: 256) */
  maxWindowSize?: number;
  /** Target window size in characters (default: 192) */
  targetWindowSize?: number;
  /** Sentence delimiter characters (default: ['。', '！', '？', '!', '?', '\n']) */
  sentenceEndings?: string[];
  /** Custom bracket pairs [open, close] */
  bracketPairs?: Array<[string, string]>;
  /** Ratio of window allocated before caret position (0.0 - 1.0, default: 0.8) */
  lookbackRatio?: number;
}

export interface ImeState {
  /** Whether Japanese IME composition is currently active */
  isComposing?: boolean;
  /** Range of unconfirmed composition text in source offsets */
  compositionRange?: { start: number; end: number } | null;
}

export interface SnappedContextWindow {
  /** Extracted clean context text slice */
  text: string;
  /** Start character index in source text (post-IME exclusion) */
  startIndex: number;
  /** End character index in source text (post-IME exclusion) */
  endIndex: number;
  /** Caret offset relative to the extracted `text` slice */
  caretOffset: number;
  /** True if start boundary was snapped to sentence or bracket boundary */
  snappedStart: boolean;
  /** True if end boundary was snapped to sentence or bracket boundary */
  snappedEnd: boolean;
  /** True if all brackets within the extracted slice are balanced */
  isBracketBalanced: boolean;
  /** Raw input text length prior to IME composition removal */
  originalLength: number;
}

export const DEFAULT_SENTENCE_ENDINGS = ['。', '！', '？', '!', '?', '\n'];

export const DEFAULT_BRACKET_PAIRS: Array<[string, string]> = [
  ['「', '」'],
  ['『', '』'],
  ['（', '）'],
  ['(', ')'],
  ['【', '】'],
  ['〈', '〉'],
  ['《', '》'],
  ['〔', '〕'],
  ['［', '］'],
  ['｛', '｝'],
  ['＜', '＞'],
  ['{', '}'],
  ['[', ']'],
  ['<', '>'],
];

interface BracketAnalysis {
  openStack: Array<{ char: string; index: number }>;
  unmatchedClose: Array<{ char: string; index: number }>;
}

export class ContextWindowSnapper {
  private readonly minWindowSize: number;
  private readonly maxWindowSize: number;
  private readonly targetWindowSize: number;
  private readonly sentenceEndings: Set<string>;
  private readonly bracketPairs: Array<[string, string]>;
  private readonly openToCloseMap: Map<string, string>;
  private readonly closeToOpenMap: Map<string, string>;
  private readonly lookbackRatio: number;

  constructor(options: ContextWindowSnapperOptions = {}) {
    this.minWindowSize = options.minWindowSize ?? 128;
    this.maxWindowSize = options.maxWindowSize ?? 256;
    this.targetWindowSize = options.targetWindowSize ?? 192;
    this.lookbackRatio = options.lookbackRatio ?? 0.8;

    if (this.minWindowSize > this.maxWindowSize) {
      this.minWindowSize = this.maxWindowSize;
    }
    if (this.targetWindowSize < this.minWindowSize) {
      this.targetWindowSize = this.minWindowSize;
    }
    if (this.targetWindowSize > this.maxWindowSize) {
      this.targetWindowSize = this.maxWindowSize;
    }

    const endings = options.sentenceEndings ?? DEFAULT_SENTENCE_ENDINGS;
    this.sentenceEndings = new Set(endings);

    this.bracketPairs = options.bracketPairs ?? DEFAULT_BRACKET_PAIRS;
    this.openToCloseMap = new Map();
    this.closeToOpenMap = new Map();

    for (const [open, close] of this.bracketPairs) {
      this.openToCloseMap.set(open, close);
      this.closeToOpenMap.set(close, open);
    }
  }

  /**
   * Excludes unconfirmed IME composition range from raw text and adjusts caret index.
   */
  public excludeComposition(
    text: string,
    caretIndex: number,
    imeState?: ImeState
  ): { cleanText: string; adjustedCaret: number } {
    if (!imeState || !imeState.isComposing || !imeState.compositionRange) {
      return {
        cleanText: text,
        adjustedCaret: Math.max(0, Math.min(text.length, caretIndex)),
      };
    }

    const { start, end } = imeState.compositionRange;
    const compStart = Math.max(0, Math.min(text.length, start));
    const compEnd = Math.max(compStart, Math.min(text.length, end));

    const cleanText = text.slice(0, compStart) + text.slice(compEnd);

    let adjustedCaret = caretIndex;
    if (caretIndex < compStart) {
      adjustedCaret = caretIndex;
    } else if (caretIndex >= compEnd) {
      adjustedCaret = caretIndex - (compEnd - compStart);
    } else {
      adjustedCaret = compStart;
    }

    adjustedCaret = Math.max(0, Math.min(cleanText.length, adjustedCaret));

    return { cleanText, adjustedCaret };
  }

  /**
   * Checks if all brackets in the text slice are properly opened and closed.
   */
  public isBracketBalanced(text: string): boolean {
    const analysis = this.analyzeBrackets(text, 0, text.length);
    return analysis.openStack.length === 0 && analysis.unmatchedClose.length === 0;
  }

  /**
   * Extracts a context window around caret position with sentence boundary & bracket snapping.
   */
  public snap(
    text: string,
    caretIndex: number,
    imeState?: ImeState
  ): SnappedContextWindow {
    const originalLength = text.length;
    const { cleanText, adjustedCaret } = this.excludeComposition(text, caretIndex, imeState);
    const totalLen = cleanText.length;

    // Handle text shorter than minimum window size
    if (totalLen <= this.minWindowSize) {
      const isBalanced = this.isBracketBalanced(cleanText);
      return {
        text: cleanText,
        startIndex: 0,
        endIndex: totalLen,
        caretOffset: adjustedCaret,
        snappedStart: false,
        snappedEnd: false,
        isBracketBalanced: isBalanced,
        originalLength,
      };
    }

    // Determine initial window range around caret
    let lookback = Math.round(this.targetWindowSize * this.lookbackRatio);
    let initStart = Math.max(0, adjustedCaret - lookback);
    let initEnd = Math.min(totalLen, initStart + this.targetWindowSize);

    // Adjust if caret falls near end or start
    if (initEnd - initStart < this.minWindowSize) {
      initStart = Math.max(0, initEnd - this.minWindowSize);
      initEnd = Math.min(totalLen, initStart + this.minWindowSize);
    }

    let start = initStart;
    let end = initEnd;
    let snappedStart = false;
    let snappedEnd = false;

    // 1. Sentence Boundary Snapping
    // --- Start Boundary ---
    if (start > 0) {
      let foundSentenceStart = -1;
      // Search backward for sentence ending starting from `start`
      for (let i = start; i >= 0; i--) {
        if (this.sentenceEndings.has(cleanText[i])) {
          foundSentenceStart = i + 1; // start immediately after sentence ending
          break;
        }
      }

      if (foundSentenceStart !== -1 && end - foundSentenceStart <= this.maxWindowSize) {
        start = foundSentenceStart;
        snappedStart = true;
      } else {
        // Forward search if backward search exceeded max window
        for (let i = start; i < adjustedCaret; i++) {
          if (this.sentenceEndings.has(cleanText[i])) {
            const candStart = i + 1;
            if (end - candStart >= this.minWindowSize) {
              start = candStart;
              snappedStart = true;
            }
            break;
          }
        }
      }
    }

    // --- End Boundary ---
    if (end < totalLen) {
      let foundSentenceEnd = -1;
      // Search forward for sentence ending
      for (let i = end - 1; i < totalLen; i++) {
        if (i >= 0 && this.sentenceEndings.has(cleanText[i])) {
          foundSentenceEnd = i + 1; // include sentence ending character
          break;
        }
      }

      if (foundSentenceEnd !== -1 && foundSentenceEnd - start <= this.maxWindowSize) {
        end = foundSentenceEnd;
        snappedEnd = true;
      } else {
        // Backward search if forward search exceeded max window
        for (let i = end - 1; i >= adjustedCaret; i--) {
          if (this.sentenceEndings.has(cleanText[i])) {
            const candEnd = i + 1;
            if (candEnd - start >= this.minWindowSize) {
              end = candEnd;
              snappedEnd = true;
            }
            break;
          }
        }
      }
    }

    // 2. Bracket Consistency Snapping
    let bracketAnalysis = this.analyzeBrackets(cleanText, start, end);

    // Resolve unmatched closing brackets (close bracket inside window, open bracket before window)
    if (bracketAnalysis.unmatchedClose.length > 0) {
      for (const closeItem of bracketAnalysis.unmatchedClose) {
        const matchingOpenChar = this.closeToOpenMap.get(closeItem.char);
        if (!matchingOpenChar) continue;

        // Search backward in text for matching open bracket
        let openIdx = -1;
        for (let i = closeItem.index - 1; i >= 0; i--) {
          if (cleanText[i] === matchingOpenChar) {
            openIdx = i;
            break;
          }
        }

        if (openIdx !== -1 && openIdx < start && end - openIdx <= this.maxWindowSize) {
          start = openIdx;
          snappedStart = true;
        } else {
          // If expanding is not possible, shift start past the close bracket
          const candStart = closeItem.index + 1;
          if (candStart <= adjustedCaret && end - candStart >= this.minWindowSize) {
            start = candStart;
            snappedStart = true;
          }
        }
      }
    }

    // Re-analyze after start adjustments
    bracketAnalysis = this.analyzeBrackets(cleanText, start, end);

    // Resolve unmatched opening brackets (open bracket inside window, close bracket after window)
    if (bracketAnalysis.openStack.length > 0) {
      // Process open brackets from right to left (outermost/latest)
      for (let k = bracketAnalysis.openStack.length - 1; k >= 0; k--) {
        const openItem = bracketAnalysis.openStack[k];
        const matchingCloseChar = this.openToCloseMap.get(openItem.char);
        if (!matchingCloseChar) continue;

        // Search forward in text for matching close bracket
        let closeIdx = -1;
        for (let i = openItem.index + 1; i < totalLen; i++) {
          if (cleanText[i] === matchingCloseChar) {
            closeIdx = i;
            break;
          }
        }

        if (closeIdx !== -1 && closeIdx >= end && closeIdx + 1 - start <= this.maxWindowSize) {
          end = closeIdx + 1;
          snappedEnd = true;
        } else {
          // If expanding is not possible, trim end back before the open bracket
          const candEnd = openItem.index;
          if (candEnd >= adjustedCaret && candEnd - start >= this.minWindowSize) {
            end = candEnd;
            snappedEnd = true;
          }
        }
      }
    }

    // Final sanity check for caret containment
    if (start > adjustedCaret) start = adjustedCaret;
    if (end < adjustedCaret) end = adjustedCaret;

    const extractedText = cleanText.slice(start, end);
    const caretOffset = adjustedCaret - start;
    const isBracketBalanced = this.isBracketBalanced(extractedText);

    return {
      text: extractedText,
      startIndex: start,
      endIndex: end,
      caretOffset,
      snappedStart,
      snappedEnd,
      isBracketBalanced,
      originalLength,
    };
  }

  /**
   * Analyzes bracket structure within range [start, end) of text.
   */
  private analyzeBrackets(text: string, start: number, end: number): BracketAnalysis {
    const openStack: Array<{ char: string; index: number }> = [];
    const unmatchedClose: Array<{ char: string; index: number }> = [];

    for (let i = start; i < end; i++) {
      const ch = text[i];

      if (this.openToCloseMap.has(ch)) {
        openStack.push({ char: ch, index: i });
      } else if (this.closeToOpenMap.has(ch)) {
        const expectedOpen = this.closeToOpenMap.get(ch);
        if (openStack.length > 0 && openStack[openStack.length - 1].char === expectedOpen) {
          openStack.pop();
        } else {
          unmatchedClose.push({ char: ch, index: i });
        }
      }
    }

    return { openStack, unmatchedClose };
  }
}
