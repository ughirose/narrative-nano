import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { CharTokenizer } from '../tokenizer/CharTokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Narrative-Nano 5.8M ONNX Model Verification', () => {
  const modelPath = path.resolve(__dirname, '../../dist/models/narrative_nano_5_8m.onnx');

  it('should verify ONNX model artifact exists on disk with valid size', () => {
    expect(fs.existsSync(modelPath)).toBe(true);
    const stat = fs.statSync(modelPath);
    // Model should be approximately 19MB (5.8M parameters in FP32)
    expect(stat.size).toBeGreaterThan(15 * 1024 * 1024);
    expect(stat.size).toBeLessThan(25 * 1024 * 1024);
  });

  it('should load ONNX session and verify input/output tensor signature', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    expect(session).toBeDefined();
    expect(session.inputNames).toEqual(['input_ids']);
    expect(session.outputNames).toEqual(['modality', 'offset', 'label', 'case', 'epistemic']);

    await session.release();
  });

  it('should run multi-task inference and produce correct tensor dimensions', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const seqLen = 128;
    const inputIds = new BigInt64Array(seqLen);
    // Fill dummy tokens
    for (let i = 0; i < seqLen; i++) {
      inputIds[i] = BigInt(i % 100);
    }

    const inputTensor = new ort.Tensor('int64', inputIds, [1, seqLen]);
    const feeds = { input_ids: inputTensor };

    const results = await session.run(feeds);

    // Verify all 5 multi-task heads
    expect(results.modality).toBeDefined();
    expect(results.modality.dims).toEqual([1, seqLen, 2]);

    expect(results.offset).toBeDefined();
    expect(results.offset.dims).toEqual([1, seqLen, 65]);

    expect(results.label).toBeDefined();
    expect(results.label.dims).toEqual([1, seqLen, 8]);

    expect(results.case).toBeDefined();
    expect(results.case.dims).toEqual([1, seqLen, 10]);

    expect(results.epistemic).toBeDefined();
    expect(results.epistemic.dims).toEqual([1, seqLen, 1]);

    // Check Epistemic POV sigmoid range [0, 1]
    const epistemicData = results.epistemic.data as Float32Array;
    for (let i = 0; i < seqLen; i++) {
      expect(epistemicData[i]).toBeGreaterThanOrEqual(0.0);
      expect(epistemicData[i]).toBeLessThanOrEqual(1.0);
    }

    await session.release();
  });

  it('should differentiate dialogue vs narrative modality on real Japanese sentences', async () => {
    const modelBuffer = fs.readFileSync(modelPath);
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
    });

    const tokenizer = new CharTokenizer();

    // 1. Dialogue sentence
    const dialogueText = '「私はその人を常に先生と呼んでいた。」';
    const diaTokens = tokenizer.encode(dialogueText, { addBos: false, addEos: false }).tokens;
    const diaInput = new BigInt64Array(128);
    for (let i = 0; i < diaTokens.length; i++) {
      diaInput[i] = BigInt(diaTokens[i]);
    }

    const diaResults = await session.run({
      input_ids: new ort.Tensor('int64', diaInput, [1, 128]),
    });
    const diaModData = diaResults.modality.data as Float32Array;
    // Logit for class 1 (dialogue) at character positions
    // Softmax for first few non-bracket tokens: exp(l1) / (exp(l0) + exp(l1))
    let diaScoreSum = 0;
    for (let i = 1; i < diaTokens.length - 1; i++) {
      const l0 = diaModData[i * 2 + 0];
      const l1 = diaModData[i * 2 + 1];
      const prob1 = Math.exp(l1) / (Math.exp(l0) + Math.exp(l1));
      diaScoreSum += prob1;
    }
    const avgDiaScore = diaScoreSum / (diaTokens.length - 2);

    // 2. Narrative sentence
    const narrativeText = '下人は羅生門の下で雨やみを待っていた。';
    const narTokens = tokenizer.encode(narrativeText, { addBos: false, addEos: false }).tokens;
    const narInput = new BigInt64Array(128);
    for (let i = 0; i < narTokens.length; i++) {
      narInput[i] = BigInt(narTokens[i]);
    }

    const narResults = await session.run({
      input_ids: new ort.Tensor('int64', narInput, [1, 128]),
    });
    const narModData = narResults.modality.data as Float32Array;
    let narScoreSum = 0;
    for (let i = 0; i < narTokens.length; i++) {
      const l0 = narModData[i * 2 + 0];
      const l1 = narModData[i * 2 + 1];
      const prob1 = Math.exp(l1) / (Math.exp(l0) + Math.exp(l1));
      narScoreSum += prob1;
    }
    const avgNarScore = narScoreSum / narTokens.length;

    // Dialogue text should have significantly higher dialogue modality probability than narrative
    expect(avgDiaScore).toBeGreaterThan(0.7);
    expect(avgNarScore).toBeLessThan(0.3);

    await session.release();
  });
});
