import { describe, it, expect } from 'vitest';
import { AozoraTextNormalizer } from '../tokenizer/AozoraTextNormalizer.js';

describe('AozoraTextNormalizer', () => {
  it('should extract escaped rubies and strip tags with clean mapping', () => {
    const raw = '彼は｜硝子戸《ガラスど》の［＃「硝子戸」に傍点］中から外を見た。';
    const result = AozoraTextNormalizer.normalize(raw);

    expect(result.cleanText).toBe('彼は硝子戸の中から外を見た。');
    expect(result.rubies.length).toBe(1);
    expect(result.rubies[0].kanji).toBe('硝子戸');
    expect(result.rubies[0].ruby).toBe('ガラスど');
    expect(result.rubies[0].cleanStart).toBe(2);
    expect(result.rubies[0].cleanEnd).toBe(5);

    // Verify offset mapping
    const cleanKanji = result.cleanText.slice(result.rubies[0].cleanStart, result.rubies[0].cleanEnd);
    expect(cleanKanji).toBe('硝子戸');
  });

  it('should extract simple rubies without pipe prefix', () => {
    const raw = '下人は羅生門《らしょうもん》の下で雨やみを待っていた。';
    const result = AozoraTextNormalizer.normalize(raw);

    expect(result.cleanText).toBe('下人は羅生門の下で雨やみを待っていた。');
    expect(result.rubies.length).toBe(1);
    expect(result.rubies[0].kanji).toBe('羅生門');
    expect(result.rubies[0].ruby).toBe('らしょうもん');
  });

  it('should expand kanji repetition marks (踊り字: 々)', () => {
    const text = '時々、山々の木々が揺れる。';
    const expanded = AozoraTextNormalizer.expandRepetitionMarks(text);
    expect(expanded).toBe('時時、山山の木木が揺れる。');
  });
});
