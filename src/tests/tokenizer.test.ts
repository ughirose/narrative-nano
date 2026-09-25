import { describe, it, expect, beforeEach } from 'vitest';
import { CharTokenizer, VOCAB_SIZE, SPECIAL_TOKENS } from '../tokenizer/CharTokenizer';

describe('CharTokenizer', () => {
  let tokenizer: CharTokenizer;

  beforeEach(() => {
    tokenizer = new CharTokenizer();
  });

  describe('Vocabulary Specification', () => {
    it('should have a fixed vocabulary size of exactly 2,048', () => {
      expect(VOCAB_SIZE).toBe(2048);
      expect(CharTokenizer.VOCAB_SIZE).toBe(2048);
      expect(tokenizer.getVocabSize()).toBe(2048);
      expect(tokenizer.getVocabTable()).toHaveLength(2048);
      expect(tokenizer.getVocab().size).toBe(2048);
    });

    it('should assign special tokens to fixed indices at the beginning of the vocabulary', () => {
      const vocabTable = tokenizer.getVocabTable();
      expect(vocabTable[0]).toBe(SPECIAL_TOKENS.PAD);
      expect(vocabTable[1]).toBe(SPECIAL_TOKENS.UNK);
      expect(vocabTable[2]).toBe(SPECIAL_TOKENS.BOS);
      expect(vocabTable[3]).toBe(SPECIAL_TOKENS.EOS);
      expect(vocabTable[4]).toBe(SPECIAL_TOKENS.MASK);
      expect(vocabTable[5]).toBe(SPECIAL_TOKENS.CLS);
      expect(vocabTable[6]).toBe(SPECIAL_TOKENS.SEP);

      expect(tokenizer.getTokenId(SPECIAL_TOKENS.PAD)).toBe(0);
      expect(tokenizer.getTokenId(SPECIAL_TOKENS.UNK)).toBe(1);
      expect(tokenizer.getTokenId(SPECIAL_TOKENS.BOS)).toBe(2);
      expect(tokenizer.getTokenId(SPECIAL_TOKENS.EOS)).toBe(3);
    });

    it('should contain Hiragana, Katakana, Punctuation/Symbols, ASCII and Jōyō/Literary Kanji', () => {
      const vocab = tokenizer.getVocab();
      // Hiragana
      expect(vocab.has('あ')).toBe(true);
      expect(vocab.has('ん')).toBe(true);
      // Katakana
      expect(vocab.has('ア')).toBe(true);
      expect(vocab.has('ン')).toBe(true);
      // Punctuation / 約物
      expect(vocab.has('、')).toBe(true);
      expect(vocab.has('。')).toBe(true);
      expect(vocab.has('「')).toBe(true);
      expect(vocab.has('」')).toBe(true);
      expect(vocab.has('『')).toBe(true);
      expect(vocab.has('』')).toBe(true);
      expect(vocab.has('…')).toBe(true);
      expect(vocab.has('―')).toBe(true);
      // ASCII
      expect(vocab.has('A')).toBe(true);
      expect(vocab.has('z')).toBe(true);
      expect(vocab.has('0')).toBe(true);
      // Common & Literary Kanji
      expect(vocab.has('私')).toBe(true);
      expect(vocab.has('蓮')).toBe(true);
      expect(vocab.has('凛')).toBe(true);
      expect(vocab.has('葵')).toBe(true);
    });
  });

  describe('Bidirectional Encoding & Decoding', () => {
    it('should encode and decode standard Japanese literary text without loss', () => {
      const text = '「吾輩は猫である。名前はまだ無かった。」';
      const result = tokenizer.encode(text);

      expect(result.tokens).toHaveLength(text.length);
      const decoded = tokenizer.decode(result.tokens);
      expect(decoded).toBe(text);
    });

    it('should support addBos and addEos options during encoding', () => {
      const text = '吾輩は猫である。';
      const result = tokenizer.encode(text, { addBos: true, addEos: true });

      expect(result.tokens[0]).toBe(CharTokenizer.BOS_TOKEN_ID);
      expect(result.tokens[result.tokens.length - 1]).toBe(CharTokenizer.EOS_TOKEN_ID);

      const decodedWithSpecial = tokenizer.decode(result.tokens, true);
      expect(decodedWithSpecial).toContain(SPECIAL_TOKENS.BOS);
      expect(decodedWithSpecial).toContain(SPECIAL_TOKENS.EOS);

      const decodedNormal = tokenizer.decode(result.tokens, false);
      expect(decodedNormal).toBe(text);
    });
  });

  describe('Editor Offset <-> Tensor Index Bidirectional Conversion', () => {
    it('should accurately convert UTF-16 and CodePoint offsets for standard characters', () => {
      const text = '風の歌を聴け';
      const { tokens, offsets } = tokenizer.encode(text);

      expect(offsets).toHaveLength(text.length);

      for (let i = 0; i < text.length; i++) {
        expect(offsets[i].tensorIndex).toBe(i);
        expect(offsets[i].utf16Start).toBe(i);
        expect(offsets[i].utf16End).toBe(i + 1);
        expect(offsets[i].codePointStart).toBe(i);
        expect(offsets[i].codePointEnd).toBe(i + 1);
        expect(offsets[i].originalChar).toBe(text[i]);
      }

      // utf16OffsetToTensorIndex
      expect(tokenizer.utf16OffsetToTensorIndex(0, offsets)).toBe(0);
      expect(tokenizer.utf16OffsetToTensorIndex(2, offsets)).toBe(2);

      // tensorIndexToUtf16Offset
      expect(tokenizer.tensorIndexToUtf16Offset(2, offsets)).toEqual({ start: 2, end: 3 });

      // codePointOffsetToTensorIndex
      expect(tokenizer.codePointOffsetToTensorIndex(3, offsets)).toBe(3);

      // tensorIndexToCodePointOffset
      expect(tokenizer.tensorIndexToCodePointOffset(3, offsets)).toEqual({ start: 3, end: 4 });
    });

    it('should correctly handle surrogate pairs (e.g. 𠮷野家) in UTF-16 vs CodePoint offsets', () => {
      const text = '𠮷野家'; // '𠮷' is a surrogate pair (2 UTF-16 code units, 1 CodePoint)
      const { tokens, offsets } = tokenizer.encode(text);

      expect(tokens).toHaveLength(3); // 3 tokens
      expect(offsets).toHaveLength(3);

      // 1st token: '𠮷'
      expect(offsets[0].originalChar).toBe('𠮷');
      expect(offsets[0].utf16Start).toBe(0);
      expect(offsets[0].utf16End).toBe(2); // 2 UTF-16 code units
      expect(offsets[0].codePointStart).toBe(0);
      expect(offsets[0].codePointEnd).toBe(1); // 1 CodePoint

      // 2nd token: '野'
      expect(offsets[1].originalChar).toBe('野');
      expect(offsets[1].utf16Start).toBe(2);
      expect(offsets[1].utf16End).toBe(3);
      expect(offsets[1].codePointStart).toBe(1);
      expect(offsets[1].codePointEnd).toBe(2);

      // 3rd token: '家'
      expect(offsets[2].originalChar).toBe('家');
      expect(offsets[2].utf16Start).toBe(3);
      expect(offsets[2].utf16End).toBe(4);
      expect(offsets[2].codePointStart).toBe(2);
      expect(offsets[2].codePointEnd).toBe(3);

      // Offset conversion tests for surrogate pair position
      expect(tokenizer.utf16OffsetToTensorIndex(0, offsets)).toBe(0);
      expect(tokenizer.utf16OffsetToTensorIndex(1, offsets)).toBe(0); // mid-surrogate pair maps to token 0
      expect(tokenizer.utf16OffsetToTensorIndex(2, offsets)).toBe(1); // '野' starts at UTF-16 offset 2

      expect(tokenizer.codePointOffsetToTensorIndex(0, offsets)).toBe(0);
      expect(tokenizer.codePointOffsetToTensorIndex(1, offsets)).toBe(1);
      expect(tokenizer.codePointOffsetToTensorIndex(2, offsets)).toBe(2);
    });
  });

  describe('OOV Projection & Zero Undefined Words Guarantee', () => {
    it('should map variant kanji to canonical vocabulary forms', () => {
      const textWithVariants = '髙木さんと﨑山さん';
      const { tokens, offsets } = tokenizer.encode(textWithVariants);

      expect(offsets[0].originalChar).toBe('髙');
      expect(offsets[0].tokenChar).toBe('高');
      expect(offsets[0].isOov).toBe(true);

      expect(offsets[5].originalChar).toBe('﨑');
      expect(offsets[5].tokenChar).toBe('崎');
      expect(offsets[5].isOov).toBe(true);

      const decoded = tokenizer.decode(tokens);
      expect(decoded).toBe('高木さんと崎山さん');
    });

    it('should normalize halfwidth katakana to fullwidth katakana', () => {
      const halfwidth = 'ｶﾀｶﾅ';
      const { tokens, offsets } = tokenizer.encode(halfwidth);

      expect(offsets[0].tokenChar).toBe('カ');
      expect(offsets[1].tokenChar).toBe('タ');
      expect(offsets[2].tokenChar).toBe('カ');
      expect(offsets[3].tokenChar).toBe('ナ');

      const decoded = tokenizer.decode(tokens);
      expect(decoded).toBe('カタカナ');
    });

    it('should safely fall back for completely unknown characters without undefined tokens or throwing', () => {
      const textWithEmoji = 'こんにちは世界🛸';
      const { tokens, offsets } = tokenizer.encode(textWithEmoji);

      expect(tokens).toBeDefined();
      expect(tokens.length).toBe(8);

      const emojiOffset = offsets[offsets.length - 1];
      expect(emojiOffset.originalChar).toBe('🛸');
      expect(emojiOffset.tokenId).toBe(CharTokenizer.UNK_TOKEN_ID);
      expect(emojiOffset.isOov).toBe(true);

      // Guarantee all tokens are within [0, 2047]
      for (const id of tokens) {
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThan(VOCAB_SIZE);
        expect(Number.isInteger(id)).toBe(true);
      }
    });

    it('should guarantee zero undefined token IDs for any arbitrary input string', () => {
      const arbitraryString = 'ABCxyz 123！？™©®¿¡λµπΩ§¶†‡';
      const { tokens } = tokenizer.encode(arbitraryString);

      for (const id of tokens) {
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThan(VOCAB_SIZE);
        expect(id).not.toBeNaN();
        expect(id).not.toBeUndefined();
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty text gracefully', () => {
      const { tokens, offsets } = tokenizer.encode('');
      expect(tokens).toEqual([]);
      expect(offsets).toEqual([]);
      expect(tokenizer.decode([])).toBe('');
    });

    it('should handle out of range offset requests gracefully', () => {
      const { offsets } = tokenizer.encode('テスト');
      expect(tokenizer.utf16OffsetToTensorIndex(-5, offsets)).toBe(0);
      expect(tokenizer.utf16OffsetToTensorIndex(100, offsets)).toBe(2);
      expect(tokenizer.codePointOffsetToTensorIndex(-5, offsets)).toBe(0);
      expect(tokenizer.codePointOffsetToTensorIndex(100, offsets)).toBe(2);
    });
  });
});
