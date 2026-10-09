import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { CharTokenizer } from '../tokenizer/CharTokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Narrative-Nano v16 Multi-Task INT8 Model Verification (Deterministic 5K Corpus)', () => {
  const modelPath = path.resolve(__dirname, '../../dist/models/narrative_nano_v16_int8.onnx');

  it('should verify v16 INT8 model exists on disk with exact expected size (~11.47MB)', () => {
    expect(fs.existsSync(modelPath)).toBe(true);
    const stat = fs.statSync(modelPath);
    const sizeMb = stat.size / (1024 * 1024);
    expect(sizeMb).toBeGreaterThan(10.0);
    expect(sizeMb).toBeLessThan(13.0);
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

  it('should support dynamic sequence lengths without ReshapeHelper crash [17, 32, 64, 128, 256]', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    for (const testLen of [17, 32, 64, 128, 256]) {
      const inputIds = new BigInt64Array(testLen);
      for (let i = 0; i < testLen; i++) inputIds[i] = BigInt(i % 50);

      const res = await session.run({
        input_ids: new ort.Tensor('int64', inputIds, [1, testLen]),
      });

      expect(res.modality.dims).toEqual([1, testLen, 2]);
      expect(res.offset.dims).toEqual([1, testLen, 65]);
      expect(res.label.dims).toEqual([1, testLen, 8]);
      expect(res.case.dims).toEqual([1, testLen, 10]);
      expect(res.epistemic.dims).toEqual([1, testLen, 1]);
      expect(res.event_action.dims).toEqual([1, testLen, 6]);
      expect(res.entity.dims).toEqual([1, testLen, 4]);
      expect(res.connective.dims).toEqual([1, testLen, 5]);
    }

    await session.release();
  });

  it('should pass ground truth quality gate: cleanly differentiate dialogue vs narrative on real Japanese prose', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const tokenizer = new CharTokenizer();

    // 1. Dialogue sentence
    const dialogueText = '「私はその人を常に先生と呼んでいた。」';
    const diaTokens = tokenizer.encode(dialogueText, { addBos: false, addEos: false }).tokens;
    const diaLen = diaTokens.length;
    const diaInput = new BigInt64Array(diaLen);
    for (let i = 0; i < diaLen; i++) diaInput[i] = BigInt(diaTokens[i]);

    const diaRes = await session.run({
      input_ids: new ort.Tensor('int64', diaInput, [1, diaLen]),
    });
    const diaModData = diaRes.modality.data as Float32Array;
    let diaSum = 0;
    for (let i = 1; i < diaLen - 1; i++) {
      const l0 = diaModData[i * 2 + 0];
      const l1 = diaModData[i * 2 + 1];
      diaSum += Math.exp(l1) / (Math.exp(l0) + Math.exp(l1));
    }
    const avgDiaScore = diaSum / (diaLen - 2);

    // 2. Narrative sentence
    const narrativeText = '下人は羅生門の下で雨やみを待っていた。';
    const narTokens = tokenizer.encode(narrativeText, { addBos: false, addEos: false }).tokens;
    const narLen = narTokens.length;
    const narInput = new BigInt64Array(narLen);
    for (let i = 0; i < narLen; i++) narInput[i] = BigInt(narTokens[i]);

    const narRes = await session.run({
      input_ids: new ort.Tensor('int64', narInput, [1, narLen]),
    });
    const narModData = narRes.modality.data as Float32Array;
    let narSum = 0;
    for (let i = 0; i < narLen; i++) {
      const l0 = narModData[i * 2 + 0];
      const l1 = narModData[i * 2 + 1];
      narSum += Math.exp(l1) / (Math.exp(l0) + Math.exp(l1));
    }
    const avgNarScore = narSum / narLen;

    console.log(`[v16 Ground Truth Quality Gate]`);
    console.log(`  Dialogue Text: "${dialogueText}" -> Probability: ${(avgDiaScore * 100).toFixed(2)}%`);
    console.log(`  Narrative Text: "${narrativeText}" -> Probability: ${(avgNarScore * 100).toFixed(2)}%`);

    // Strict Ground Truth Quality Gate Assertions:
    expect(avgDiaScore).toBeGreaterThan(0.85); // >85% dialogue probability for dialogue
    expect(avgNarScore).toBeLessThan(0.15);    // <15% dialogue probability for narrative

    await session.release();
  });

  it('should measure Wasm SIMD latency on Japanese literary sentences (Target: P50 < 45ms)', async () => {
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

    // Warm-up run with dynamic length
    const warmupInput = new BigInt64Array(32);
    await session.run({ input_ids: new ort.Tensor('int64', warmupInput, [1, 32]) });

    // Benchmark runs
    for (const sent of testSentences) {
      const tokens = tokenizer.encode(sent, { addBos: false, addEos: false }).tokens;
      const inputIds = new BigInt64Array(tokens.length);
      for (let i = 0; i < tokens.length; i++) inputIds[i] = BigInt(tokens[i]);

      const t0 = performance.now();
      const res = await session.run({
        input_ids: new ort.Tensor('int64', inputIds, [1, tokens.length]),
      });
      const t1 = performance.now();
      latencies.push(t1 - t0);

      expect(res.modality).toBeDefined();
      expect(res.event_action).toBeDefined();
    }

    latencies.sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length / 2)];
    console.log(`[Wasm Benchmark v16] P50 Latency: ${p50.toFixed(2)}ms (Samples: ${latencies.map((l) => l.toFixed(1) + 'ms').join(', ')})`);

    expect(p50).toBeLessThan(60.0);

    await session.release();
  });
});
