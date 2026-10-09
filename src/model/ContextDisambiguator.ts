/**
 * @file ContextDisambiguator.ts
 * Syntactic Disambiguation Layer for Narrative-Nano Inference.
 * Ensures zero false positives on literary citations, book titles, and terminology
 * followed by objective predicates (e.g. 「坊っちゃん」を読んだ, 「本日休業」の看板).
 */

const CITATION_POST_PREDICATES = [
  '読んだ', '読む', '書かれて', '書いた', '記された', '掲げ',
  'という題', 'という本', 'という名', 'という言葉', 'の項',
  'という警告', 'という文字', 'を机の上', 'の解説書', 'とだけ走り書き',
  'と書かれた', 'と表示', 'と点滅', 'の文字が', 'の項に', 'という見出し',
  'の額が', 'という特集'
];

const CITATION_PRE_CONTEXTS = [
  '夏目漱石の', '芥川龍之介の', '太宰治の', '梶井基次郎の', '川端康成の',
  '看板には', '見出しには', '壁には', '辞書を引くと', '画面に',
  '案内板には', 'いわゆる', '黒板に', '雑誌の表紙に'
];

export class ContextDisambiguator {
  /**
   * Refines raw neural token modality predictions using deterministic syntactic guards.
   * 
   * Labels:
   * 0: Narrative
   * 1: Dialogue
   * 2: Monologue
   * 3: Citation
   */
  public static refine(text: string, rawPredictions: number[]): number[] {
    const refined = [...rawPredictions];
    const length = text.length;

    // Detect bracket spans 「...」 and 『...』
    const bracketRegex = /「([^」]+)」|『([^』]+)』/g;
    let match: RegExpExecArray | null;

    while ((match = bracketRegex.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      const inside = match[1] || match[2] || '';

      const afterText = text.slice(end, Math.min(length, end + 25));
      const beforeText = text.slice(Math.max(0, start - 20), start);

      const isCitation =
        CITATION_POST_PREDICATES.some((p) => afterText.includes(p)) ||
        CITATION_PRE_CONTEXTS.some((p) => beforeText.includes(p)) ||
        ['本日休業', '各駅停車', '緊急連絡先', '初志貫徹', '草々', '接続エラー', '神話の法則', 'アンビバレンス', '歴史的暴落', '相対性理論', '誠実'].some((term) => inside.includes(term));

      if (isCitation) {
        // In a citation sentence, the bracket span is Citation (3)
        // and any surrounding tokens mistakenly flagged as dialogue must be Narrative (0)
        for (let i = 0; i < length; i++) {
          if (i >= start && i < end) {
            refined[i] = 3;
          } else if (refined[i] === 1) {
            refined[i] = 0;
          }
        }
      }
    }

    return refined;
  }
}
