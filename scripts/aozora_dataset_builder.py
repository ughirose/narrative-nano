#!/usr/bin/env python3
"""
narrative-nano/scripts/aozora_dataset_builder.py

Automated Aozora Bunko Text Retrieval, Ruby Handling, Normalization,
and Structured Dataset Construction for Narrative-Nano.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple, Any

# Canonical URLs for Public Domain Masterpieces from Aozora Bunko
AOZORA_WORKS = [
    {
        "title": "こころ",
        "author": "夏目漱石",
        "url": "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/773_ruby_5968/773_ruby_5968.txt"
    },
    {
        "title": "坊っちゃん",
        "author": "夏目漱石",
        "url": "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/752_ruby_2438/752_ruby_2438.txt"
    },
    {
        "title": "羅生門",
        "author": "芥川龍之介",
        "url": "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/127_ruby_150/127_ruby_150.txt"
    },
    {
        "title": "走れメロス",
        "author": "太宰治",
        "url": "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1567_ruby_4948/1567_ruby_4948.txt"
    }
]

# 10 Japanese Case Particles Definition
CASE_ORDER = ["が", "を", "に", "で", "と", "より", "から", "まで", "へ", "の"]
CASE_MAP = {
    "が": {"case_id": 0, "name": "主格", "katakana": "ガ"},
    "を": {"case_id": 1, "name": "対格", "katakana": "ヲ"},
    "に": {"case_id": 2, "name": "与格", "katakana": "ニ"},
    "で": {"case_id": 3, "name": "具格", "katakana": "デ"},
    "と": {"case_id": 4, "name": "共格", "katakana": "ト"},
    "より": {"case_id": 5, "name": "奪格", "katakana": "ヨリ"},
    "から": {"case_id": 6, "name": "起点格", "katakana": "カラ"},
    "まで": {"case_id": 7, "name": "終点格", "katakana": "マデ"},
    "へ": {"case_id": 8, "name": "方向格", "katakana": "ヘ"},
    "の": {"case_id": 9, "name": "連体修飾格", "katakana": "ノ"},
}

INTERNAL_STATE_KEYWORDS = [
    "思っ", "思わ", "思い", "感じ", "考え", "悔し", "悲し", "嬉し", "驚い", "怖",
    "怒り", "望ん", "疑っ", "悩み", "迷っ", "信じ", "焦っ", "悟っ", "知っ", "気絶"
]

CONJECTURAL_KEYWORDS = [
    "ようだ", "ようだった", "らしい", "かもしれ", "に違いない", "のだろ", "気がした", "ようにも見え"
]


class AozoraTextCleaner:
    """Handles ruby notation, annotations, and text normalization."""

    @staticmethod
    def strip_aozora_headers_footers(raw_text: str) -> str:
        """Strips Aozora header metadata, separator bars, and footer publication info."""
        lines = raw_text.splitlines()
        content_lines = []
        in_header = True
        dash_count = 0

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("-------"):
                dash_count += 1
                if dash_count == 2:
                    in_header = False
                continue
            if in_header and dash_count < 2:
                continue

            # Check footer start
            if line_stripped.startswith("底本：") or line_stripped.startswith("［＃本文終わり］"):
                break

            content_lines.append(line)

        # If header separator wasn't found, use standard text fallback
        cleaned = "\n".join(content_lines) if content_lines else raw_text
        return cleaned

    @staticmethod
    def process_ruby_and_markup(text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Processes Aozora ruby notations:
        1. ｜親文字《るび》 -> 親文字
        2. 漢字《るび》 -> 漢字
        Strips ［＃...］ editorial tags and ［＃「...」に傍点］.
        Returns cleaned text and extracted ruby spans.
        """
        rubies = []

        # Remove editorial notes ［＃...］
        clean_stage1 = re.sub(r'［＃[^］]*］', '', text)

        # Regex for escaped ruby: ｜([^《]+)《([^》]+)》
        def repl_escaped(match):
            kanji = match.group(1)
            ruby = match.group(2)
            rubies.append({"kanji": kanji, "ruby": ruby, "type": "escaped"})
            return kanji

        clean_stage2 = re.sub(r'｜([^《\n]+)《([^》\n]+)》', repl_escaped, clean_stage1)

        # Regex for simple ruby: ([一-龠々〆ヵヶ]+)《([^》]+)》
        def repl_simple(match):
            kanji = match.group(1)
            ruby = match.group(2)
            rubies.append({"kanji": kanji, "ruby": ruby, "type": "simple"})
            return kanji

        clean_stage3 = re.sub(r'([一-龠々〆ヵヶ]+)《([^》\n]+)》', repl_simple, clean_stage2)

        # Remove dangling ruby brackets if any remain
        clean_stage4 = re.sub(r'《[^》\n]*》', '', clean_stage3)
        clean_stage5 = clean_stage4.replace('｜', '')

        # Unicode normalization (NFKC) while preserving Japanese quotation marks
        normalized = unicodedata.normalize('NFKC', clean_stage5)

        # Normalize repetition marks (踊り字) if required
        normalized = AozoraTextCleaner.expand_repetition_marks(normalized)

        return normalized, rubies

    @staticmethod
    def expand_repetition_marks(text: str) -> str:
        """Expands repetition mark 々 where preceded by kanji."""
        chars = list(text)
        result = []
        for i, c in enumerate(chars):
            if c == '々' and i > 0 and ('\u4e00' <= chars[i - 1] <= '\u9fff'):
                result.append(chars[i - 1])
            else:
                result.append(c)
        return "".join(result)


class NovelTextSegmenter:
    """Segments literary novel text into sentence boundaries, conversation/dialogue, and narrative."""

    DIALOGUE_PATTERN = re.compile(r'(「[^」]*」|『[^』]*』)')

    @classmethod
    def segment_text(cls, raw_text: str) -> List[Dict[str, Any]]:
        """
        Splits novel text into sentence units and classifies each as dialogue or narrative.
        Returns a list of dictionaries with segmented text and type.
        """
        results = []
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        for line in lines:
            tokens = cls.DIALOGUE_PATTERN.split(line)
            for token in tokens:
                token = token.strip()
                if not token:
                    continue

                if (token.startswith('「') and token.endswith('」')) or (token.startswith('『') and token.endswith('』')):
                    results.append({
                        "text": token,
                        "text_type": "dialogue",
                        "is_dialogue": True,
                        "raw_dialogue": token[1:-1]
                    })
                else:
                    sentences = cls.split_sentences(token)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            results.append({
                                "text": sentence,
                                "text_type": "narrative",
                                "is_dialogue": False,
                                "raw_dialogue": None
                            })

        return results

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Splits narrative Japanese text by sentence boundaries (。, ！, ？, !)."""
        pattern = re.compile(r'([^。！？!]*[。！？!])')
        matches = pattern.findall(text)
        remainder = pattern.sub('', text)

        sentences = [m.strip() for m in matches if m.strip()]
        if remainder.strip():
            sentences.append(remainder.strip())
        return sentences


class JapaneseCaseAnnotator:
    """Annotates 10 case particles and syntactic head/offset hints."""

    PARTICLE_ORDER = ["より", "から", "まで", "が", "を", "に", "で", "と", "へ", "の"]
    PARTICLE_SPLIT_REGEX = re.compile(r'[、。！？「」『』\s]|(?:より|から|まで|が|を|に|で|と|へ|の)')

    @classmethod
    def analyze_particles(cls, sentence: str) -> List[Dict[str, Any]]:
        detected = []
        n = len(sentence)
        i = 0

        while i < n:
            matched = False
            for particle in cls.PARTICLE_ORDER:
                plen = len(particle)
                if sentence[i:i + plen] == particle:
                    case_info = CASE_MAP[particle]
                    start_span = i
                    end_span = i + plen

                    target_start = max(0, start_span - 15)
                    before_sub = sentence[target_start:start_span]
                    sub_tokens = [tok for tok in cls.PARTICLE_SPLIT_REGEX.split(before_sub) if tok]
                    target_word = sub_tokens[-1] if sub_tokens else sentence[max(0, start_span - 4):start_span]

                    detected.append({
                        "particle": particle,
                        "case_id": case_info["case_id"],
                        "case_name": case_info["name"],
                        "case_katakana": case_info["katakana"],
                        "span": [start_span, end_span],
                        "target_noun": target_word
                    })
                    i += plen
                    matched = True
                    break

            if not matched:
                i += 1

        return detected

    @classmethod
    def detect_epistemic_pov(cls, sentence: str, is_dialogue: bool) -> float:
        """Computes epistemic/POV internal state score [0.0 - 1.0]."""
        if is_dialogue:
            # Dialogue has overt speaker modality
            return 0.1

        score = 0.0
        # Check internal state keywords
        has_internal = any(kw in sentence for kw in INTERNAL_STATE_KEYWORDS)
        has_conjectural = any(kw in sentence for kw in CONJECTURAL_KEYWORDS)

        if has_internal:
            score += 0.8
        if has_conjectural:
            # Conjectural marker mitigates unauthorized internal state
            score = max(0.2, score * 0.5)

        if "私" in sentence or "僕" in sentence or "俺" in sentence:
            score = max(score, 0.5)

        return min(1.0, score)


class AozoraDatasetBuilder:
    """Fetches Aozora masterpieces, processes rubies, segments, and outputs structured JSONL."""

    def __init__(self, works: Optional[List[Dict[str, str]]] = None):
        self.works = works or AOZORA_WORKS

    def fetch_work_text(self, work_info: Dict[str, str]) -> str:
        """Fetches text content from URL with utf-8 or shift_jis decoding."""
        url = work_info["url"]
        title = work_info["title"]
        print(f"[*] Fetching: {title} ({work_info['author']})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        # Try shift_jis then utf-8
        try:
            return data.decode("shift_jis")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")

    def build_dataset(self, max_samples_per_work: Optional[int] = None) -> List[Dict[str, Any]]:
        """Builds structured dataset from all configured works."""
        dataset = []
        global_idx = 0

        for work in self.works:
            title = work["title"]
            author = work["author"]
            try:
                raw_text = self.fetch_work_text(work)
            except Exception as e:
                print(f"[!] Warning: Failed to fetch {title}: {e}. Skipping.", file=sys.stderr)
                continue

            # Strip headers / footers
            body_text = AozoraTextCleaner.strip_aozora_headers_footers(raw_text)

            # Process ruby and markup
            cleaned_text, rubies = AozoraTextCleaner.process_ruby_and_markup(body_text)

            # Segment into units
            segments = NovelTextSegmenter.segment_text(cleaned_text)
            print(f"[+] {title}: extracted {len(segments)} segments (clean text length: {len(cleaned_text):,} chars).")

            count = 0
            for seg in segments:
                text = seg["text"]
                # Keep meaningful lengths between 4 and 128 characters
                if len(text) < 4 or len(text) > 128:
                    continue

                is_dialogue = seg["is_dialogue"]
                particles = JapaneseCaseAnnotator.analyze_particles(text)
                epistemic_score = JapaneseCaseAnnotator.detect_epistemic_pov(text, is_dialogue)

                record = {
                    "id": f"aozora_{global_idx:07d}",
                    "work_title": title,
                    "author": author,
                    "text": text,
                    "text_type": seg["text_type"],
                    "is_dialogue": is_dialogue,
                    "length": len(text),
                    "case_particles": particles,
                    "epistemic_score": round(epistemic_score, 4),
                    "rubies_count": len([r for r in rubies if r["kanji"] in text]),
                }
                dataset.append(record)
                global_idx += 1
                count += 1
                if max_samples_per_work and count >= max_samples_per_work:
                    break

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Aozora Bunko Literary Dataset Builder for Narrative-Nano")
    parser.add_argument("--output", "-o", default="dist/datasets/aozora_literary_dataset.jsonl", help="Output path for JSONL dataset")
    parser.add_argument("--max-per-work", "-m", type=int, default=None, help="Max sentences per work (default: all)")
    args = parser.parse_args()

    builder = AozoraDatasetBuilder()
    print("[*] Starting Aozora Bunko Dataset Build Pipeline...")
    dataset = builder.build_dataset(max_samples_per_work=args.max_per_work)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Built dataset with {len(dataset):,} structured records.")
    print(f"[+] Output saved to: {args.output}")


if __name__ == "__main__":
    main()
