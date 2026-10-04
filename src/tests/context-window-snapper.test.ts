import { describe, it, expect } from 'vitest';
import { ContextWindowSnapper } from '../tokenizer/ContextWindowSnapper.js';

describe('ContextWindowSnapper', () => {
  it('should extract a context window around caret within default min/max bounds', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 30,
      maxWindowSize: 80,
      targetWindowSize: 50,
      lookbackRatio: 0.8,
    });

    const prefix = '昔々あるところにおじいさんとおばあさんが住んでいました。';
    const body = 'おじいさんは山へ芝刈りに、おばあさんは川へ洗濯に行きました。';
    const suffix = '川から大きな桃がどんぶらこどんぶらこと流れてきました。';
    const text = prefix + body + suffix;

    const caretIndex = prefix.length + 15; // In the body
    const result = snapper.snap(text, caretIndex);

    expect(result.text.length).toBeGreaterThanOrEqual(30);
    expect(result.text.length).toBeLessThanOrEqual(80);
    expect(result.caretOffset).toBeGreaterThanOrEqual(0);
    expect(result.caretOffset).toBeLessThanOrEqual(result.text.length);
    expect(text.slice(result.startIndex, result.endIndex)).toBe(result.text);
  });

  it('should snap start and end boundaries to sentence endings (句点、！？、改行)', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 20,
      maxWindowSize: 70,
      targetWindowSize: 40,
      lookbackRatio: 0.5,
    });

    const text = '第一文です。第二文は少し長めの文章になります！第三文はどうでしょうか？第四文の終わり。第五文もあります。第六文まで続きます。';
    const caretIndex = 25; // Inside 第二文

    const result = snapper.snap(text, caretIndex);

    // Start boundary should snap after '第一文です。'
    expect(result.snappedStart).toBe(true);
    expect(result.text.startsWith('第二文')).toBe(true);

    // End boundary should snap after a sentence ending character ('！' or '？' or '。')
    expect(result.snappedEnd).toBe(true);
    expect(/[！？。]$/.test(result.text)).toBe(true);
  });

  it('should snap start boundary backward when slice starts with unmatched close bracket', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 20,
      maxWindowSize: 100,
      targetWindowSize: 40,
      lookbackRatio: 0.8,
    });

    const text = '冒頭文。 「この台詞はかぎ括弧で囲まれています」 文末。';
    // Position caret inside quote so naive initial window would start inside the quote after '「'
    const caretIndex = text.indexOf('台詞');

    const result = snapper.snap(text, caretIndex);

    expect(result.isBracketBalanced).toBe(true);
    expect(result.text.includes('「')).toBe(true);
    expect(result.text.includes('」')).toBe(true);
  });

  it('should snap end boundary forward to include closing bracket for unclosed opening bracket', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 15,
      maxWindowSize: 80,
      targetWindowSize: 25,
      lookbackRatio: 0.8,
    });

    const text = '前文。 「非常に長い会話の途中にキャレットがあります」 後文。';
    const caretIndex = text.indexOf('会話');

    const result = snapper.snap(text, caretIndex);

    expect(result.isBracketBalanced).toBe(true);
    expect(result.text.includes('「')).toBe(true);
    expect(result.text.includes('」')).toBe(true);
  });

  it('should handle nested brackets (『』 inside 「」) without bracket corruption', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 30,
      maxWindowSize: 120,
      targetWindowSize: 60,
    });

    const text = '彼が「あの本『銀河鉄道の夜』は本当に素晴らしかった」と感銘を受けて語った。';
    const caretIndex = text.indexOf('銀河鉄道');

    const result = snapper.snap(text, caretIndex);

    expect(result.isBracketBalanced).toBe(true);
    expect(result.text).toContain('『銀河鉄道の夜』');
  });

  it('should exclude Japanese IME composition range when active and adjust caret offset', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 20,
      maxWindowSize: 100,
      targetWindowSize: 50,
    });

    const rawText = '吾輩は猫である。なまえは[未確定IME入力]まだない。';
    const compStart = rawText.indexOf('[未確定IME入力]');
    const compEnd = compStart + '[未確定IME入力]'.length;
    const caretIndex = compStart + 3; // Caret inside unconfirmed composition

    const result = snapper.snap(rawText, caretIndex, {
      isComposing: true,
      compositionRange: { start: compStart, end: compEnd },
    });

    expect(result.text).not.toContain('[未確定IME入力]');
    expect(result.originalLength).toBe(rawText.length);
    // Clean text: '吾輩は猫である。なまえはまだない。'
    expect(result.text).toContain('吾輩は猫である。なまえはまだない。');
  });

  it('should create atomic slice immediately upon IME commit (isComposing: false)', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 20,
      maxWindowSize: 100,
      targetWindowSize: 50,
    });

    const confirmedText = '吾輩は猫である。名前はまだない。';
    const caretIndex = confirmedText.indexOf('まだない');

    const result = snapper.snap(confirmedText, caretIndex, {
      isComposing: false,
      compositionRange: null,
    });

    expect(result.text).toBe(confirmedText);
    expect(result.isBracketBalanced).toBe(true);
  });

  it('should handle short text gracefully by returning full clean text', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 128,
      maxWindowSize: 256,
      targetWindowSize: 192,
    });

    const shortText = '短文です。';
    const result = snapper.snap(shortText, 2);

    expect(result.text).toBe(shortText);
    expect(result.startIndex).toBe(0);
    expect(result.endIndex).toBe(shortText.length);
    expect(result.caretOffset).toBe(2);
    expect(result.snappedStart).toBe(false);
    expect(result.snappedEnd).toBe(false);
  });

  it('should handle text with custom options and custom bracket pairs', () => {
    const snapper = new ContextWindowSnapper({
      minWindowSize: 10,
      maxWindowSize: 40,
      targetWindowSize: 20,
      sentenceEndings: ['\n'],
      bracketPairs: [['<', '>']],
    });

    const text = 'Line 1\nLine 2 <custom context>\nLine 3';
    const caretIndex = text.indexOf('custom');

    const result = snapper.snap(text, caretIndex);

    expect(result.isBracketBalanced).toBe(true);
    expect(result.text).toContain('<custom context>');
  });
});
