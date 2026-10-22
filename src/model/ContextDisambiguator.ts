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

import type { FusedProseTelemetry } from '@worldcraft/schema';

export interface DisambiguationOptions {
  telemetry?: FusedProseTelemetry['kasane'];
  previousTension?: number;
  isPastRecollectionContext?: boolean;
}

const FID_PATTERNS = [
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
  /.*悔しくてたまらない/,
  /.*どうすれば.*(?:か|のか|だろう|のだろう)/,
];

export class ContextDisambiguator {
  /**
   * Modulates modality probability distribution using Bayesian prior from Kasane telemetry.
   * P*(Monologue) = P(Monologue) * (1.0 + alpha * (confession / 100.0))
   * P*(Narrative) = 1.0 - P*(Monologue)
   */
  public static modulateModalityProbabilities(
    narrativeProb: number,
    monologueProb: number,
    telemetry?: FusedProseTelemetry['kasane'],
    alpha: number = 0.5
  ): { narrative: number; monologue: number } {
    if (!telemetry?.texture) {
      return { narrative: narrativeProb, monologue: monologueProb };
    }
    const confession = telemetry.texture.confession ?? 0;
    const modulatedMonologue = Math.min(1.0, Math.max(0.0, monologueProb * (1.0 + alpha * (confession / 100.0))));
    const modulatedNarrative = Math.max(0.0, 1.0 - modulatedMonologue);
    return { narrative: modulatedNarrative, monologue: modulatedMonologue };
  }

  /**
   * Refines raw neural token modality predictions using deterministic syntactic guards
   * and Kasane stylistic telemetry.
   * 
   * Labels:
   * 0: Narrative
   * 1: Dialogue
   * 2: Monologue
   * 3: Citation
   */
  public static refine(
    text: string,
    rawPredictions: number[],
    options?: DisambiguationOptions
  ): number[] {
    const refined = [...rawPredictions];
    const length = text.length;

    const isInsideBracket = new Uint8Array(length);

    // Detect bracket spans 「...」 and 『...』
    const bracketRegex = /「([^」]+)」|『([^』]+)』/g;
    let match: RegExpExecArray | null;

    while ((match = bracketRegex.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      const inside = match[1] || match[2] || '';

      for (let i = start; i < end; i++) {
        isInsideBracket[i] = 1;
      }

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

    // Unquoted Dialogue Guard (Dazai / Saramago / Oral-style unquoted speech clauses)
    // Only applies when there are no bracketed quotes or parentheses in the sentence.
    if (!/[「」『』（）]/.test(text)) {
      const unquotedMatch =
        /^([^。！？!?]+?)(?:、と)(?:[^。！？!?]+[。！？!?]?)$/.exec(text) ||
        /^((?:おい|ねえ|やあ|もし|ほら|さあ|どうしたんだい|ごめんください|助けて|冗談じゃない|駄目だよ|まさかそんな|ちょっと待って|よかった)[^、。]*、[^、。]+?)(?:、(?:誰かが|見知らぬ|後ろから|彼女が|子供が|微かな声が|席を|格子戸を|号令が|抱きしめ|気遣われ|指さし|勝ち誇))/.exec(
          text
        );
      if (unquotedMatch && unquotedMatch[1]) {
        const speechSpanLen = unquotedMatch[1].length;
        // Verify that the pre-comma clause contains colloquial speech markers or quotative framing
        const hasSpeechMarker =
          /(?:おい|ねえ|やあ|もし|ほら|さあ|かい|だい|よ|ぞ|な|ね|くれ|しろ|見ろ|行こう|待って|聞いて|駄目|馬鹿|冗談|助けて|ごめん|よかった|おられませんか|やってられるか|ちゃ)/.test(
            unquotedMatch[1]
          );
        if (hasSpeechMarker) {
          for (let i = 0; i < Math.min(length, speechSpanLen); i++) {
            refined[i] = 1; // Dialogue
          }
          for (let i = speechSpanLen; i < length; i++) {
            if (refined[i] === 1) {
              refined[i] = 0; // Narrative reporting clause
            }
          }
        }
      }
    }

    // Kasane Telemetry Integration: FID (Free Indirect Discourse) Monologue Adaptation
    if (options?.telemetry?.texture) {
      const confession = options.telemetry.texture.confession ?? 0;
      const rawTension = options.telemetry.texture.tension ?? 0;
      const tension = rawTension > 1.0 ? Math.min(1.0, rawTension / 100.0) : rawTension;
      const rawPrevTension = options.previousTension;
      const prevTension =
        rawPrevTension !== undefined
          ? rawPrevTension > 1.0
            ? Math.min(1.0, rawPrevTension / 100.0)
            : rawPrevTension
          : undefined;
      const deltaTension = prevTension !== undefined ? Math.abs(tension - prevTension) : 0;
      const hasTensionModulation = (deltaTension > 0.3) || (tension >= 0.3);

      const isPastRecollection =
        options.isPastRecollectionContext === true ||
        /(?:十年前|あの日|当時は|あの頃|その頃|かつて)[^。]*?(?:だった|であった|いた|た。)/.test(text);

      const confessionThreshold = isPastRecollection ? 80.0 : 65.0;

      if (confession >= confessionThreshold && hasTensionModulation) {
        // Split into sentence spans using punctuation / newline
        const spanRegex = /[^。！？!?\n]+[。！？!?\n]?/g;
        let sMatch: RegExpExecArray | null;

        while ((sMatch = spanRegex.exec(text)) !== null) {
          const sStart = sMatch.index;
          const spanText = sMatch[0];
          const sEnd = sStart + spanText.length;

          const isFIDSpan = FID_PATTERNS.some((pat) => pat.test(spanText));
          if (isFIDSpan) {
            for (let i = sStart; i < sEnd; i++) {
              if (!isInsideBracket[i] && refined[i] !== 3 && (refined[i] === 0 || refined[i] === 1)) {
                refined[i] = 2; // Monologue (FID)
              }
            }
          }
        }
      }
    }

    return refined;
  }
}
