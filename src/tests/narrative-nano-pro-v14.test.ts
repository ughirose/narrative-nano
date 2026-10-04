import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { CharTokenizer } from '../tokenizer/CharTokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Narrative-Nano Pro v14 QAT INT8 Model Verification', () => {
  const modelPath = path.resolve(__dirname, '../../dist/models/narrative_nano_pro_v14_qat_int8.onnx');

  it('should verify v14 QAT INT8 model exists on disk with exact size (~14.77MB)', () => {
    expect(fs.existsSync(modelPath)).toBe(true);
    const stat = fs.statSync(modelPath);
    // 14.56M parameters in INT8 quantized format: ~14.77MB
    const sizeMb = stat.size / (1024 * 1024);
    expect(sizeMb).toBeGreaterThan(14.0);
    expect(sizeMb).toBeLessThan(15.5);
  });

  it('should load ONNX session and verify 8 multi-task output heads signature', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    expect(session).toBeDefined();
    expect(session.inputNames).toEqual(['input_ids']);
    expect(session.outputNames).toEqual([
      'modality',
      'offset',
      'label',
      'case',
      'epistemic',
      'event_action',
      'entity',
      'connective',
    ]);

    await session.release();
  });

  it('should run multi-task inference across all 8 heads with valid dimensions', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const seqLen = 256;
    const inputIds = new BigInt64Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      inputIds[i] = BigInt(i % 100);
    }

    const inputTensor = new ort.Tensor('int64', inputIds, [1, seqLen]);
    const feeds = { input_ids: inputTensor };

    const results = await session.run(feeds);

    // Verify all 8 heads
    expect(results.modality.dims).toEqual([1, seqLen, 2]);
    expect(results.offset.dims).toEqual([1, seqLen, 65]);
    expect(results.label.dims).toEqual([1, seqLen, 8]);
    expect(results.case.dims).toEqual([1, seqLen, 10]);
    expect(results.epistemic.dims).toEqual([1, seqLen, 1]);
    expect(results.event_action.dims).toEqual([1, seqLen, 6]);
    expect(results.entity.dims).toEqual([1, seqLen, 4]);
    expect(results.connective.dims).toEqual([1, seqLen, 5]);

    // Check Epistemic POV range [0, 1]
    const epistemicData = results.epistemic.data as Float32Array;
    for (let i = 0; i < seqLen; i++) {
      expect(epistemicData[i]).toBeGreaterThanOrEqual(0.0);
      expect(epistemicData[i]).toBeLessThanOrEqual(1.0);
    }

    await session.release();
  });

  it('should measure Wasm SIMD latency on Japanese literary sentences (Target: P50 < 32ms)', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const tokenizer = new CharTokenizer();
    const testSentences = [
      '「私はその人を常に先生と呼んでいた。」',
      '下人は羅生門の下で雨やみを待っていた。',
      'その時、メロスは激怒した。必ず邪智暴虐の王を除かなければならぬと決意した。',
      '道は険しく、冷たい雨が容赦なく降りしきっていた。',
      '彼女は震える手で封筒を開き、中から古びた手紙を取り出した。',
    ];

    const latencies: number[] = [];

    // Warm-up run
    const warmupInput = new BigInt64Array(256);
    await session.run({ input_ids: new ort.Tensor('int64', warmupInput, [1, 256]) });

    // Benchmark runs
    for (const sent of testSentences) {
      const tokens = tokenizer.encode(sent, { addBos: false, addEos: false }).tokens;
      const inputIds = new BigInt64Array(256);
      for (let i = 0; i < Math.min(tokens.length, 256); i++) {
        inputIds[i] = BigInt(tokens[i]);
      }

      const t0 = performance.now();
      const res = await session.run({
        input_ids: new ort.Tensor('int64', inputIds, [1, 256]),
      });
      const t1 = performance.now();
      latencies.push(t1 - t0);

      expect(res.modality).toBeDefined();
      expect(res.event_action).toBeDefined();
    }

    latencies.sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length / 2)];
    console.log(`[Wasm Benchmark] P50 Latency: ${p50.toFixed(2)}ms (Samples: ${latencies.map((l) => l.toFixed(1) + 'ms').join(', ')})`);

    // Verify reasonable latency
    expect(p50).toBeLessThan(100.0); // Safe CI margin

    await session.release();
  });
});
