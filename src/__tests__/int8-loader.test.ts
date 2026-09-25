import { describe, it, expect } from 'vitest';
import { Int8ModelLoader } from '../loader/int8-loader.js';

describe('Int8ModelLoader', () => {
  it('should initialize with default options', () => {
    const loader = new Int8ModelLoader();
    expect(loader.isLoaded()).toBe(false);
    expect(loader.getMetadata().inputNames).toEqual([]);
  });

  it('should correctly quantize Float32Array to Int8Array', () => {
    const floatData = new Float32Array([0.0, 0.5, 1.0, -0.5, -1.0]);
    const scale = 0.01;
    const zeroPoint = 0;

    const quantized = Int8ModelLoader.quantizeFloat32ToInt8(floatData, scale, zeroPoint);

    expect(quantized.length).toBe(floatData.length);
    expect(quantized[0]).toBe(0);
    expect(quantized[1]).toBe(50);
    expect(quantized[2]).toBe(100);
    expect(quantized[3]).toBe(-50);
    expect(quantized[4]).toBe(-100);
  });

  it('should correctly dequantize Int8Array to Float32Array', () => {
    const int8Data = new Int8Array([0, 50, 100, -50, -100]);
    const scale = 0.01;
    const zeroPoint = 0;

    const dequantized = Int8ModelLoader.dequantizeInt8ToFloat32(int8Data, scale, zeroPoint);

    expect(dequantized.length).toBe(int8Data.length);
    expect(dequantized[0]).toBeCloseTo(0.0);
    expect(dequantized[1]).toBeCloseTo(0.5);
    expect(dequantized[2]).toBeCloseTo(1.0);
    expect(dequantized[3]).toBeCloseTo(-0.5);
    expect(dequantized[4]).toBeCloseTo(-1.0);
  });

  it('should clamp quantized values to Int8 range [-128, 127]', () => {
    const floatData = new Float32Array([100.0, -100.0]);
    const scale = 0.1;
    const quantized = Int8ModelLoader.quantizeFloat32ToInt8(floatData, scale, 0);

    expect(quantized[0]).toBe(127);
    expect(quantized[1]).toBe(-128);
  });

  it('should create ORT INT8 and Float32 tensors', () => {
    const int8Data = new Int8Array([1, 2, 3, 4]);
    const float32Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);

    const tensorInt8 = Int8ModelLoader.createInt8Tensor(int8Data, [2, 2]);
    const tensorFloat = Int8ModelLoader.createFloat32Tensor(float32Data, [2, 2]);

    expect(tensorInt8.type).toBe('int8');
    expect(tensorFloat.type).toBe('float32');
    expect(tensorInt8.dims).toEqual([2, 2]);
  });
});
