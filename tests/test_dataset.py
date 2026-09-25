#!/usr/bin/env python3
"""
tests/test_dataset.py

Unit tests for Japanese Literary 10-Case Synthetic Dataset Generator and Pipeline.
Tests sentence segmentation, dialogue/narrative splitting, 10-case particle analysis,
subject-predicate twisting detection/generation, and INT8 validation dataset extraction.
"""

import json
import os
import sys
import tempfile
import unittest

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_synthetic_cases import (
    NovelTextSegmenter,
    JapaneseCaseAnalyzer,
    SubjectPredicateTwistEngine,
    GeminiSyntheticGenerator,
    SyntheticDatasetPipeline,
    CASE_PARTICLES
)


class TestNovelTextSegmenter(unittest.TestCase):
    """Test literary novel text segmentation and dialogue/narrative classification."""

    def test_split_sentences(self):
        text = "主人公は立ち上がった。闇の中で不気味な音が響く！誰だ？"
        sentences = NovelTextSegmenter.split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "主人公は立ち上がった。")
        self.assertEqual(sentences[1], "闇の中で不気味な音が響く！")
        self.assertEqual(sentences[2], "誰だ？")

    def test_segment_text_dialogue_and_narrative(self):
        raw_text = "月明かりが回廊を照らしていた。「そこを動くな！」少年は剣を構えた。『伝説の聖剣』が輝く。"
        segmented = NovelTextSegmenter.segment_text(raw_text)

        # Check dialogue vs narrative types
        dialogues = [s for s in segmented if s["text_type"] == "dialogue"]
        narratives = [s for s in segmented if s["text_type"] == "narrative"]

        self.assertGreaterEqual(len(dialogues), 2)
        self.assertGreaterEqual(len(narratives), 2)
        self.assertEqual(dialogues[0]["text"], "「そこを動くな！」")
        self.assertEqual(dialogues[1]["text"], "『伝説の聖剣』")


class TestJapaneseCaseAnalyzer(unittest.TestCase):
    """Test 10-case particle analysis and tagging."""

    def test_10_case_particles_coverage(self):
        # Sentence containing all 10 case particles (ガ, ヲ, ニ, デ, ト, ヨリ, カラ, マデ, ヘ, ノ)
        test_sentence = "主人公が剣を手に持って部屋で仲間と共に古い峠より街から城まで北へ向かい、王の部屋に入った。"
        detected = JapaneseCaseAnalyzer.analyze_particles(test_sentence)

        detected_cases = {item["case_name"] for item in detected}

        expected_cases = {"主格", "対格", "与格", "具格", "共格", "奪格", "起点格", "終点格", "方向格", "連体修飾格"}
        for case in expected_cases:
            self.assertIn(case, detected_cases, f"Expected case {case} in detected particles.")

    def test_particle_spans_and_targets(self):
        sentence = "彼が街へ走った。"
        detected = JapaneseCaseAnalyzer.analyze_particles(sentence)
        self.assertEqual(len(detected), 2)

        ga_particle = next(p for p in detected if p["particle"] == "が")
        self.assertEqual(ga_particle["case_katakana"], "ガ")
        self.assertEqual(ga_particle["target"], "彼")
        self.assertEqual(ga_particle["span"], [1, 2])

        he_particle = next(p for p in detected if p["particle"] == "へ")
        self.assertEqual(he_particle["case_katakana"], "ヘ")
        self.assertEqual(he_particle["target"], "街")
        self.assertEqual(he_particle["span"], [3, 4])


class TestSubjectPredicateTwistEngine(unittest.TestCase):
    """Test generation and detection of subject-predicate twisting."""

    def test_generate_twisted_case(self):
        twist_case = SubjectPredicateTwistEngine.generate_twisted_case(0)
        self.assertIn("twisted_sentence", twist_case)
        self.assertIn("corrected_sentence", twist_case)
        self.assertIn("twist_type", twist_case)
        self.assertNotEqual(twist_case["twisted_sentence"], twist_case["corrected_sentence"])


class TestSyntheticDatasetPipeline(unittest.TestCase):
    """Test end-to-end dataset generation and INT8 validation dataset extraction."""

    def setUp(self):
        self.pipeline = SyntheticDatasetPipeline()

    def test_generate_jsonl_dataset(self):
        dataset = self.pipeline.generate_jsonl_dataset(total_count=15)
        self.assertEqual(len(dataset), 15)

        for record in dataset:
            self.assertIn("id", record)
            self.assertIn("text", record)
            self.assertIn("text_type", record)
            self.assertIn("case_particles", record)
            self.assertIn("has_subject_predicate_twist", record)

    def test_extract_validation_dataset_5k(self):
        base_dataset = self.pipeline.generate_jsonl_dataset(total_count=10)
        val_dataset = self.pipeline.extract_validation_dataset(base_dataset, target_size=5000)

        self.assertEqual(len(val_dataset), 5000)
        self.assertEqual(val_dataset[0]["id"], "val_int8_000001")
        self.assertEqual(val_dataset[4999]["id"], "val_int8_005000")
        self.assertTrue(val_dataset[0]["metadata"]["is_validation"])
        self.assertEqual(val_dataset[0]["metadata"]["quantization_target"], "INT8")

    def test_end_to_end_file_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            val_path = os.path.join(tmpdir, "val_5k.jsonl")

            dataset = self.pipeline.generate_jsonl_dataset(total_count=20)
            val_dataset = self.pipeline.extract_validation_dataset(dataset, target_size=100)

            with open(train_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            with open(val_path, "w", encoding="utf-8") as f:
                for item in val_dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))

            # Read back and verify valid JSON
            with open(val_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 100)


if __name__ == "__main__":
    unittest.main()
