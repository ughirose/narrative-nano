/**
 * narrative-nano: AozoraTextNormalizer.ts
 *
 * Specialized Literary Text Normalizer for Aozora Bunko (青空文庫) formats.
 * Extracts ruby annotations, handles folding characters (踊り字: 々, ゝ, ゞ, 〳〵),
 * strips editorial markup (［＃...］), and maintains exact 1-to-1 character offset
 * projections back to the raw source text.
 */

export interface RubySpan {
  kanji: string;
  ruby: string;
  sourceStart: number;
  sourceEnd: number;
  cleanStart: number;
  cleanEnd: number;
}

export interface AozoraNormalizeResult {
  cleanText: string;
  rubies: RubySpan[];
  // Mapping from cleanText codePoint index to sourceText codePoint index
  cleanToSourceOffsetMap: number[];
  // Mapping from sourceText codePoint index to cleanText codePoint index (-1 if stripped)
  sourceToCleanOffsetMap: number[];
}

export class AozoraTextNormalizer {
  // Regex for Aozora ruby: ｜親文字《ルビ》 or 親文字《ルビ》
  private static readonly RUBY_WITH_ESCAPE = /｜([^《]+)《([^》]+)》/g;
  private static readonly RUBY_SIMPLE = /([一-龠々〆ヵヶ]+)《([^》]+)》/g;
  // Regex for Aozora markup tags: ［＃...］
  private static readonly AOZORA_TAG = /［＃[^］]+］/g;

  /**
   * Normalizes Aozora Bunko raw text into clean text while preserving
   * bidirectional offset maps and extracting all ruby annotations.
   */
  public static normalize(rawText: string): AozoraNormalizeResult {
    const rubies: RubySpan[] = [];
    const sourceChars = Array.from(rawText);
    const sourceLen = sourceChars.length;

    // Track characters to keep and their source positions
    const cleanChars: string[] = [];
    const cleanToSourceOffsetMap: number[] = [];
    const sourceToCleanOffsetMap: number[] = new Array(sourceLen).fill(-1);

    let i = 0;
    while (i < sourceLen) {
      // 1. Check for Aozora tag ［＃...］
      if (sourceChars[i] === '［' && i + 1 < sourceLen && sourceChars[i + 1] === '＃') {
        const closeIdx = rawText.indexOf('］', i);
        if (closeIdx !== -1) {
          // Skip the entire tag
          i = closeIdx + 1;
          continue;
        }
      }

      // 2. Check for escaped ruby: ｜親文字《ルビ》
      if (sourceChars[i] === '｜') {
        const openRuby = rawText.indexOf('《', i);
        const closeRuby = openRuby !== -1 ? rawText.indexOf('》', openRuby) : -1;
        if (openRuby !== -1 && closeRuby !== -1 && openRuby < closeRuby) {
          const kanjiPart = rawText.slice(i + 1, openRuby);
          const rubyPart = rawText.slice(openRuby + 1, closeRuby);

          const startCleanIdx = cleanChars.length;
          const kanjiChars = Array.from(kanjiPart);

          for (let k = 0; k < kanjiChars.length; k++) {
            const currentSourceIdx = i + 1 + k;
            cleanToSourceOffsetMap.push(currentSourceIdx);
            sourceToCleanOffsetMap[currentSourceIdx] = cleanChars.length;
            cleanChars.push(kanjiChars[k]);
          }

          rubies.push({
            kanji: kanjiPart,
            ruby: rubyPart,
            sourceStart: i,
            sourceEnd: closeRuby + 1,
            cleanStart: startCleanIdx,
            cleanEnd: cleanChars.length,
          });

          i = closeRuby + 1;
          continue;
        }
      }

      // 3. Check for simple ruby: 漢字《ルビ》
      if (sourceChars[i] === '《') {
        const closeRuby = rawText.indexOf('》', i);
        if (closeRuby !== -1) {
          // Look backwards in cleanChars to find contiguous kanji
          let back = cleanChars.length - 1;
          while (back >= 0 && /[一-龠々〆ヵヶ]/.test(cleanChars[back])) {
            back--;
          }
          const kanjiStartClean = back + 1;
          if (kanjiStartClean < cleanChars.length) {
            const kanji = cleanChars.slice(kanjiStartClean).join('');
            const ruby = rawText.slice(i + 1, closeRuby);
            const sourceStart = cleanToSourceOffsetMap[kanjiStartClean];

            rubies.push({
              kanji,
              ruby,
              sourceStart,
              sourceEnd: closeRuby + 1,
              cleanStart: kanjiStartClean,
              cleanEnd: cleanChars.length,
            });
          }

          i = closeRuby + 1;
          continue;
        }
      }

      // 4. Normal character
      const cleanIdx = cleanChars.length;
      cleanToSourceOffsetMap.push(i);
      sourceToCleanOffsetMap[i] = cleanIdx;
      cleanChars.push(sourceChars[i]);
      i++;
    }

    return {
      cleanText: cleanChars.join(''),
      rubies,
      cleanToSourceOffsetMap,
      sourceToCleanOffsetMap,
    };
  }

  /**
   * Expands vertical repetition marks / folding characters (踊り字) where appropriate
   * for syntactic / semantic parsing.
   */
  public static expandRepetitionMarks(text: string): string {
    const chars = Array.from(text);
    const result: string[] = [];

    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (c === '々' && i > 0 && /[一-龠]/.test(chars[i - 1])) {
        result.push(chars[i - 1]);
      } else if (c === 'ゝ' && i > 0 && /[ぁ-ん]/.test(chars[i - 1])) {
        result.push(chars[i - 1]);
      } else if (c === 'ゞ' && i > 0 && /[ぁ-ん]/.test(chars[i - 1])) {
        // Simple voiced sound approximation or preserve
        result.push(chars[i - 1]);
      } else {
        result.push(c);
      }
    }

    return result.join('');
  }
}
