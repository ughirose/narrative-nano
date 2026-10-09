import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { CharTokenizer } from '../tokenizer/CharTokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface BenchmarkSample {
  id: string;
  text: string;
  gold_category: string;
  subcategory: string;
  has_bracket: boolean;
  expected_modality: 'dialogue' | 'narrative' | 'monologue';
}

describe('Narrative-Nano 100-Sentence Literary Empirical Benchmark', () => {
  const modelPath = path.resolve(__dirname, '../../dist/models/narrative_nano_v16_int8.onnx');
  const datasetPath = path.resolve(__dirname, '../../dist/datasets/literary_benchmark_100.jsonl');

  it('evaluates v16 ONNX model across all 6 categories and reveals true false positives / false negatives', async () => {
    expect(fs.existsSync(modelPath)).toBe(true);
    expect(fs.existsSync(datasetPath)).toBe(true);

    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const tokenizer = new CharTokenizer();
    const rawLines = fs.readFileSync(datasetPath, 'utf-8').trim().split('\n');
    const samples: BenchmarkSample[] = rawLines.map((l) => JSON.parse(l));

    const categoryStats: Record<string, { total: number; predDialogue: number; predNarrative: number; scores: number[] }> = {};
    const falsePositives: { text: string; prob: number; category: string }[] = [];
    const falseNegatives: { text: string; prob: number; category: string }[] = [];

    for (const s of samples) {
      if (!categoryStats[s.gold_category]) {
        categoryStats[s.gold_category] = { total: 0, predDialogue: 0, predNarrative: 0, scores: [] };
      }

      const tokens = tokenizer.encode(s.text, { addBos: false, addEos: false }).tokens;
      const seqLen = tokens.length;
      const inputIds = new BigInt64Array(seqLen);
      for (let i = 0; i < seqLen; i++) inputIds[i] = BigInt(tokens[i]);

      const res = await session.run({
        input_ids: new ort.Tensor('int64', inputIds, [1, seqLen]),
      });

      const modData = res.modality.data as Float32Array;
      let diaScoreSum = 0;
      for (let i = 0; i < seqLen; i++) {
        const l0 = modData[i * 2 + 0];
        const l1 = modData[i * 2 + 1];
        const p1 = Math.exp(l1) / (Math.exp(l0) + Math.exp(l1));
        diaScoreSum += p1;
      }
      const avgDiaProb = diaScoreSum / seqLen;

      categoryStats[s.gold_category].total += 1;
      categoryStats[s.gold_category].scores.push(avgDiaProb);

      if (avgDiaProb >= 0.5) {
        categoryStats[s.gold_category].predDialogue += 1;
        if (s.gold_category === 'narrative' || s.gold_category === 'narrative_pure') {
          falsePositives.push({ text: s.text, prob: avgDiaProb, category: s.gold_category });
        }
      } else {
        categoryStats[s.gold_category].predNarrative += 1;
        if (s.gold_category === 'dialogue' || s.gold_category === 'mixed' || s.gold_category === 'dialogue_unquoted') {
          falseNegatives.push({ text: s.text, prob: avgDiaProb, category: s.gold_category });
        }
      }
    }

    console.log('\n========================================================================');
    console.log('         100-SENTENCE LITERARY BENCHMARK: EMPIRICAL AUDIT REPORT');
    console.log('========================================================================');
    for (const [cat, data] of Object.entries(categoryStats)) {
      const avgScore = data.scores.reduce((a, b) => a + b, 0) / data.total;
      console.log(
        `Category: ${cat.padEnd(20)} | Total: ${String(data.total).padStart(2)} | Pred Dia: ${String(data.predDialogue).padStart(2)} | Pred Nar: ${String(data.predNarrative).padStart(2)} | Avg Dia Prob: ${(avgScore * 100).toFixed(1)}%`
      );
    }

    console.log('\n--- False Positives (Narrative mistakenly classified as Dialogue) ---');
    console.log(`Count: ${falsePositives.length}`);
    for (const fp of falsePositives) {
      console.log(`  [${fp.category}] ${(fp.prob * 100).toFixed(1)}% | ${fp.text}`);
    }

    console.log('\n--- False Negatives (Dialogue mistakenly classified as Narrative) ---');
    console.log(`Count: ${falseNegatives.length}`);
    for (const fn of falseNegatives) {
      console.log(`  [${fn.category}] ${(fn.prob * 100).toFixed(1)}% | ${fn.text}`);
    }
    console.log('========================================================================\n');

    await session.release();
  });
});
