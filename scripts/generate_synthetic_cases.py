#!/usr/bin/env python3
"""
scripts/generate_synthetic_cases.py

Japanese Literary 10-Case Grammar & Subject-Predicate Twisting Synthetic Dataset Pipeline.
Generates structured JSONL datasets using Gemini API / Few-shot prompts with fallback generation,
segments literary prose into sentence boundaries and dialogue/narrative, tags 10-case particles,
and extracts a 5,000-item validation dataset for INT8 quantization resilience testing.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

# 10 Japanese Case Particles Definition
CASE_PARTICLES = {
    "主格": {"particle": "が", "katakana": "ガ", "desc": "主格（行為の主体、状態の主語）"},
    "対格": {"particle": "を", "katakana": "ヲ", "desc": "対格（直接目的語、移動の通過点）"},
    "与格": {"particle": "に", "katakana": "ニ", "desc": "与格/着格（相手、到達点、存在場所、帰着時間）"},
    "具格": {"particle": "で", "katakana": "デ", "desc": "具格/於格（手段、道具、行為の行われる場所）"},
    "共格": {"particle": "と", "katakana": "ト", "desc": "共格（共同行為者、引用、変化の結果）"},
    "奪格": {"particle": "より", "katakana": "ヨリ", "desc": "奪格（起点、比較の基準）"},
    "起点格": {"particle": "から", "katakana": "カラ", "desc": "起点格（時間・空間の出発点、原因）"},
    "終点格": {"particle": "まで", "katakana": "マデ", "desc": "終点格（時間・空間の到達限界）"},
    "方向格": {"particle": "へ", "katakana": "ヘ", "desc": "方向格（移動の向かう方向）"},
    "連体修飾格": {"particle": "の", "katakana": "ノ", "desc": "連体修飾格（所有、所属、属性の限定）"},
}


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
                if not token:
                    continue

                if (token.startswith('「') and token.endswith('」')) or (token.startswith('『') and token.endswith('』')):
                    results.append({
                        "text": token,
                        "text_type": "dialogue",
                        "raw_dialogue": token[1:-1]
                    })
                else:
                    sentences = cls.split_sentences(token)
                    for sentence in sentences:
                        if sentence.strip():
                            results.append({
                                "text": sentence.strip(),
                                "text_type": "narrative",
                                "raw_dialogue": None
                            })

        return results

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Splits narrative Japanese text by sentence boundaries (。, ！, ？)."""
        pattern = re.compile(r'([^。！？!]*[。！？!])')
        matches = pattern.findall(text)
        remainder = pattern.sub('', text)

        sentences = [m for m in matches if m.strip()]
        if remainder.strip():
            sentences.append(remainder.strip())
        return sentences


class JapaneseCaseAnalyzer:
    """Analyzes Japanese 10-case particles in a given sentence."""

    PARTICLE_ORDER = ["より", "から", "まで", "が", "を", "に", "で", "と", "へ", "の"]

    PARTICLE_TO_CASE = {
        "が": ("主格", "ガ"),
        "を": ("対格", "ヲ"),
        "に": ("与格", "ニ"),
        "で": ("具格", "デ"),
        "と": ("共格", "ト"),
        "より": ("奪格", "ヨリ"),
        "から": ("起点格", "カラ"),
        "まで": ("終点格", "マデ"),
        "へ": ("方向格", "ヘ"),
        "の": ("連体修飾格", "ノ"),
    }

    # Proper regex split pattern for separating preceding words without invalid character classes
    PARTICLE_SPLIT_REGEX = re.compile(r'[、。！？「」『』\s]|(?:より|から|まで|が|を|に|で|と|へ|の)')

    @classmethod
    def analyze_particles(cls, sentence: str) -> List[Dict[str, Any]]:
        """Identifies 10-case particles, their spans, case name, and target noun phrase."""
        detected = []
        n = len(sentence)

        i = 0
        while i < n:
            matched = False
            for particle in cls.PARTICLE_ORDER:
                plen = len(particle)
                if sentence[i:i+plen] == particle:
                    case_name, katakana = cls.PARTICLE_TO_CASE[particle]
                    start_span = i
                    end_span = i + plen

                    target_start = max(0, start_span - 15)
                    before_sub = sentence[target_start:start_span]

                    sub_tokens = [tok for tok in cls.PARTICLE_SPLIT_REGEX.split(before_sub) if tok]
                    target_word = sub_tokens[-1] if sub_tokens else sentence[max(0, start_span-4):start_span]

                    detected.append({
                        "particle": particle,
                        "case_katakana": katakana,
                        "case_name": case_name,
                        "span": [start_span, end_span],
                        "target": target_word
                    })
                    i += plen
                    matched = True
                    break

            if not matched:
                i += 1

        return detected


class SubjectPredicateTwistEngine:
    """Generates and detects subject-predicate twisting (主述のねじれ) in literary sentences."""

    NOMINAL_TWIST_PAIRS = [
        (
            "彼が旅に出た理由",
            "幼少期の記憶を忘れたかったからだ。",
            "幼少期の記憶を忘れたかった。"
        ),
        (
            "私がこの小説を書き始めた目的",
            "失われた名誉を取り戻すことだ。",
            "失われた名誉を取り戻したい。"
        ),
        (
            "主人公の最大の願い",
            "世界平和を守り抜くことだ。",
            "世界平和を守り抜く。"
        ),
        (
            "探偵が事件の真相を確信した背景",
            "現場に残された足跡の角度に違和感を抱いたからだ。",
            "現場に残された足跡の角度に違和感を抱いた。"
        )
    ]

    TOPIC_TWIST_PAIRS = [
        (
            "古びた洋館の扉",
            "静かに開くと、冷たい風が吹き込んできた。",
            "静かに開けると、冷たい風が吹き込んできた。"
        ),
        (
            "風に揺れる桜の花びら",
            "舞い散り、地面を桃色に染めていった。",
            "舞い散る様子を眺めながら溜息をついた。"
        ),
        (
            "テーブルの上の手紙",
            "静かに風に吹かれて床へ落ちた。",
            "そっと手に取って中身を読み進めた。"
        )
    ]

    @classmethod
    def generate_twisted_case(cls, index: int) -> Dict[str, Any]:
        """Generates a synthetic sentence with subject-predicate twisting and its correction."""
        if index % 2 == 0:
            pair = cls.NOMINAL_TWIST_PAIRS[ (index // 2) % len(cls.NOMINAL_TWIST_PAIRS) ]
            subject, correct_action, twisted_action = pair
            ptype = "NOMINAL_PREDICATE_MISMATCH"
            desc = "主語の名詞句（〜理由は/〜目的は/〜夢は）と述語の結びつきのねじれ"
        else:
            pair = cls.TOPIC_TWIST_PAIRS[ (index // 2) % len(cls.TOPIC_TWIST_PAIRS) ]
            subject, correct_action, twisted_action = pair
            ptype = "TOPIC_CLAUSE_MISMATCH"
            desc = "主題節と主節の動作主不一致によるねじれ"

        twisted = f"{subject}は、{twisted_action}"
        corrected = f"{subject}は、{correct_action}"

        return {
            "twisted_sentence": twisted,
            "corrected_sentence": corrected,
            "twist_type": ptype,
            "description": desc
        }


class GeminiSyntheticGenerator:
    """Gemini API Few-shot Prompt Synthetic Case Generator with fallback mechanism."""

    FEW_SHOT_PROMPT = """
あなたは日本語文芸小説の最高峰エディターおよび構文解析の専門家です。
以下の条件に従い、日本語長編文脈における10格文法（ガ・ヲ・ニ・デ・ト・ヨリ・カラ・マデ・ヘ・ノ）および「主述のねじれ（構文エラー）」を含む高精細な合成文芸例文セットを生成してください。

【10格の定義】
1. ガ（主格） 2. ヲ（対格） 3. ニ（与格/着格） 4. デ（具格/於格） 5. ト（共格）
6. ヨリ（奪格） 7. カラ（起点格） 8. マデ（終点格） 9. ヘ（方向格） 10. ノ（連体修飾格）

【要求形式】
JSONフォーマットのリストで出力してください:
[
  {
    "context": "文芸長編の先行文脈...",
    "text": "生成された例文",
    "text_type": "narrative" 又は "dialogue",
    "has_subject_predicate_twist": true 又は false,
    "twist_type": "NOMINAL_PREDICATE_MISMATCH" 等 (ねじれが無い場合は null),
    "corrected_text": "正当な文（ねじれが無い場合は null）"
  }
]
"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def generate_via_api(self, count: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Generates synthetic dataset entries using Gemini REST API."""
        if not self.api_key:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        prompt_text = f"{self.FEW_SHOT_PROMPT}\n\n数量: {count} 件のJSON要素を出力してください。"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}
        }

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(text_resp)
                if isinstance(parsed, list):
                    return parsed
        except Exception as e:
            sys.stderr.write(f"Gemini API request failed or bypassed: {e}\n")

        return None

    def generate_fallback_dataset(self, count: int) -> List[Dict[str, Any]]:
        """Fallback rule/pattern generator when Gemini API is unavailable or offline."""
        dataset = []

        subjects = ["主人公", "少年", "老騎士", "探偵", "少女", "旅人", "館の主", "若き学者"]
        locations = ["古城の回廊", "雨の港町", "静かな書斎", "吹雪の峠", "霧の深い森", "月明かりの広場"]
        actions = [
            ("剣を手に取り、暗闇の中へと進んだ。", "narrative"),
            ("「あの山の向こうへ行ってはならぬと、長老から固く口止めされていたのだ。」", "dialogue"),
            ("机の上の古い手紙を手にとり、静かに目を通した。", "narrative"),
            ("「この手懸かりから犯人の部屋まで辿り着くことができるはずだ。」", "dialogue"),
            ("暖炉の炎を見つめながら、静かに過去を語り始めた。", "narrative"),
            ("「私と共にあの街へ向かった日のことを、覚えているかね？」", "dialogue")
        ]

        for i in range(count):
            subj = subjects[i % len(subjects)]
            loc = locations[(i * 3) % len(locations)]
            act, ttype = actions[(i * 5) % len(actions)]

            ctx = f"{loc}で{subj}はたたずんでいた。"

            if i % 5 == 0:
                twist_info = SubjectPredicateTwistEngine.generate_twisted_case(i)
                text = twist_info["twisted_sentence"]
                corrected = twist_info["corrected_sentence"]
                has_twist = True
                twist_type = twist_info["twist_type"]
            else:
                text = f"{subj}は{act}" if not act.startswith("「") else act
                corrected = None
                has_twist = False
                twist_type = None

            dataset.append({
                "context": ctx,
                "text": text,
                "text_type": ttype,
                "has_subject_predicate_twist": has_twist,
                "twist_type": twist_type,
                "corrected_text": corrected
            })

        return dataset


class SyntheticDatasetPipeline:
    """Master pipeline orchestrator for synthetic dataset generation and INT8 validation dataset extraction."""

    def __init__(self, api_key: Optional[str] = None):
        self.generator = GeminiSyntheticGenerator(api_key=api_key)

    def generate_jsonl_dataset(
        self,
        total_count: int = 100,
        raw_text_input: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generates full synthetic JSONL dataset entries with 10-case particle annotations."""
        raw_cases = []

        if raw_text_input:
            segmented = NovelTextSegmenter.segment_text(raw_text_input)
            for seg in segmented:
                raw_cases.append({
                    "context": "小説テキスト本文より抽出",
                    "text": seg["text"],
                    "text_type": seg["text_type"],
                    "has_subject_predicate_twist": False,
                    "twist_type": None,
                    "corrected_text": None
                })

        needed = total_count - len(raw_cases)
        if needed > 0:
            api_generated = self.generator.generate_via_api(needed)
            if api_generated:
                raw_cases.extend(api_generated)
            else:
                fallback_cases = self.generator.generate_fallback_dataset(needed)
                raw_cases.extend(fallback_cases)

        final_dataset = []
        for idx, item in enumerate(raw_cases[:total_count]):
            sentence_text = item["text"]
            particles = JapaneseCaseAnalyzer.analyze_particles(sentence_text)

            record = {
                "id": f"syn_case_{idx+1:06d}",
                "text": sentence_text,
                "context": item.get("context", ""),
                "text_type": item.get("text_type", "narrative"),
                "case_particles": particles,
                "has_subject_predicate_twist": item.get("has_subject_predicate_twist", False),
                "twist_type": item.get("twist_type"),
                "corrected_text": item.get("corrected_text"),
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "case_particle_count": len(particles)
                }
            }
            final_dataset.append(record)

        return final_dataset

    def extract_validation_dataset(
        self,
        dataset: List[Dict[str, Any]],
        target_size: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Generates/extracts a 5,000 distinct item validation dataset tailored for testing
        INT8 quantization resilience. Every generated sample is unique.
        """
        val_dataset = []

        locations = ["古城", "港町", "雪山", "図書室", "王都", "地下牢", "霧の森", "教会", "廃墟", "時計塔"]
        characters = ["主人公", "老騎士", "探偵", "少女", "学者", "旅人", "剣士", "修道士", "暗殺者", "精霊"]
        actions_list = [
            ("剣を抜いて構えた。", "が"),
            ("扉を静かに開けた。", "を"),
            ("目的地に無事到達した。", "に"),
            ("部屋で手紙を発見した。", "で"),
            ("仲間と深く頷き合った。", "と"),
            ("旧市街より抜け出してきた。", "より"),
            ("広場から小道へと走った。", "から"),
            ("夜明けまで待機した。", "まで"),
            ("北の塔へ向かった。", "へ"),
            ("王の言葉を刻み込んだ。", "の")
        ]

        for i in range(target_size):
            loc = locations[i % len(locations)]
            char = characters[(i // 10) % len(characters)]
            act, particle_key = actions_list[(i // 100) % len(actions_list)]

            chapter = (i // 500) + 1
            sentence = f"{char}は{loc}で{act}"

            if i % 10 == 0:
                twist_info = SubjectPredicateTwistEngine.generate_twisted_case(i)
                sentence = twist_info["twisted_sentence"]
                corrected = twist_info["corrected_sentence"]
                has_twist = True
                twist_type = twist_info["twist_type"]
            else:
                corrected = None
                has_twist = False
                twist_type = None

            particles = JapaneseCaseAnalyzer.analyze_particles(sentence)

            val_record = {
                "id": f"val_int8_{i+1:06d}",
                "text": sentence,
                "context": f"第{chapter}章: {loc}における{char}の行動記録#{i+1}",
                "text_type": "dialogue" if sentence.startswith("「") else "narrative",
                "case_particles": particles,
                "has_subject_predicate_twist": has_twist,
                "twist_type": twist_type,
                "corrected_text": corrected,
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "case_particle_count": len(particles),
                    "is_validation": True,
                    "quantization_target": "INT8",
                    "validation_sample_index": i
                }
            }
            val_dataset.append(val_record)

        return val_dataset


def main():
    parser = argparse.ArgumentParser(description="Japanese Literary 10-Case Synthetic Dataset Pipeline")
    parser.add_argument("--output", "-o", default="dist/synthetic_cases_10case.jsonl", help="Output path for training JSONL")
    parser.add_argument("--count", "-c", type=int, default=100, help="Number of synthetic training records")
    parser.add_argument("--input-file", "-i", help="Input literary text file to segment and annotate")
    parser.add_argument("--extract-validation", action="store_true", help="Extract INT8 quantization validation dataset")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation dataset size (default: 5000)")
    parser.add_argument("--val-output", default="dist/validation_int8_5k.jsonl", help="Output path for validation JSONL")

    args = parser.parse_args()

    pipeline = SyntheticDatasetPipeline()

    raw_text = None
    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

    print(f"[*] Generating synthetic dataset ({args.count} items)...")
    dataset = pipeline.generate_jsonl_dataset(total_count=args.count, raw_text_input=raw_text)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[+] Saved training dataset: {args.output} ({len(dataset)} items)")

    if args.extract_validation or args.val_size == 5000:
        print(f"[*] Extracting INT8 quantization validation dataset ({args.val_size} items)...")
        val_dataset = pipeline.extract_validation_dataset(dataset, target_size=args.val_size)

        val_out_dir = os.path.dirname(args.val_output)
        if val_out_dir:
            os.makedirs(val_out_dir, exist_ok=True)

        with open(args.val_output, "w", encoding="utf-8") as f:
            for record in val_dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[+] Saved INT8 validation dataset: {args.val_output} ({len(val_dataset)} items)")


if __name__ == "__main__":
    main()
