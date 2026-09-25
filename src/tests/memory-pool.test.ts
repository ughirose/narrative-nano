import { describe, it, expect, beforeEach } from 'vitest';
import { MemoryPool } from '../runtime/MemoryPool.js';

describe('MemoryPool Static Memory Pool Allocator', () => {
  let pool: MemoryPool;

  beforeEach(() => {
    pool = new MemoryPool({
      int8Capacity: 1024,
      float32Capacity: 1024,
      int32Capacity: 512,
      uint8Capacity: 512,
      enableSimd: true,
    });
  });

  it('should initialize with correct initial stats', () => {
    const stats = pool.getStats();
    expect(stats.int8Capacity).toBe(1024);
    expect(stats.float32Capacity).toBe(1024);
    expect(stats.int32Capacity).toBe(512);
    expect(stats.uint8Capacity).toBe(512);
    expect(stats.int8Allocated).toBe(0);
    expect(stats.float32Allocated).toBe(0);
    expect(stats.totalAllocatedBytes).toBe(0);
    expect(stats.resetCount).toBe(0);
  });

  it('should acquire typed array buffers from pre-allocated pool without dynamic allocation', () => {
    const f32 = pool.acquireFloat32(128);
    const i8 = pool.acquireInt8(256);
    const i32 = pool.acquireInt32(64);
    const u8 = pool.acquireUint8(32);

    expect(f32.length).toBe(128);
    expect(i8.length).toBe(256);
    expect(i32.length).toBe(64);
    expect(u8.length).toBe(32);

    const stats = pool.getStats();
    expect(stats.float32Allocated).toBe(128);
    expect(stats.int8Allocated).toBe(256);
    expect(stats.int32Allocated).toBe(64);
    expect(stats.uint8Allocated).toBe(32);

    // 128*4 + 256*1 + 64*4 + 32*1 = 512 + 256 + 256 + 32 = 1056 bytes
    expect(stats.totalAllocatedBytes).toBe(1056);
    expect(stats.peakAllocatedBytes).toBe(1056);
  });

  it('should throw error when pool capacity is exceeded', () => {
    expect(() => pool.acquireInt8(2000)).toThrow(/MemoryPool Int8 overflow/);
    expect(() => pool.acquireFloat32(2000)).toThrow(/MemoryPool Float32 overflow/);
  });

  it('should reset allocation pointers in O(1) time without garbage collection', () => {
    const f32a = pool.acquireFloat32(500);
    f32a[0] = 42.5;

    let stats = pool.getStats();
    expect(stats.float32Allocated).toBe(500);

    pool.reset();

    stats = pool.getStats();
    expect(stats.float32Allocated).toBe(0);
    expect(stats.totalAllocatedBytes).toBe(0);
    expect(stats.resetCount).toBe(1);
    expect(stats.peakAllocatedBytes).toBe(2000); // 500 * 4 bytes

    // Re-acquire after reset reuses same memory slab
    const f32b = pool.acquireFloat32(500);
    expect(f32b[0]).toBe(42.5); // previous values intact in same buffer
  });

  it('should compute 128-bit Wasm SIMD vector dot product accurately for Float32', () => {
    const len = 100;
    const a = new Float32Array(len);
    const b = new Float32Array(len);

    for (let i = 0; i < len; i++) {
      a[i] = i * 0.1;
      b[i] = (i % 5) * 0.5;
    }

    let expected = 0.0;
    for (let i = 0; i < len; i++) {
      expected += a[i] * b[i];
    }

    // Test SIMD enabled
    const resultSimd = pool.dotProductSIMD(a, b, len);
    expect(resultSimd).toBeCloseTo(expected, 4);

    // Test non-SIMD fallback
    const noSimdPool = new MemoryPool({ enableSimd: false });
    const resultNoSimd = noSimdPool.dotProductSIMD(a, b, len);
    expect(resultNoSimd).toBeCloseTo(expected, 4);
  });

  it('should compute 128-bit Wasm SIMD vector dot product accurately for Int8', () => {
    const len = 64;
    const a = new Int8Array(len);
    const b = new Int8Array(len);

    for (let i = 0; i < len; i++) {
      a[i] = (i % 10) - 5;
      b[i] = (i % 6) - 3;
    }

    let expected = 0;
    for (let i = 0; i < len; i++) {
      expected += a[i] * b[i];
    }

    const resultSimd = pool.dotProductInt8SIMD(a, b, len);
    expect(resultSimd).toBe(expected);
  });

  it('should perform fast zero-allocation Float32 to Int8 quantization and dequantization', () => {
    const input = new Float32Array([0.0, 1.2, -1.2, 5.0, -10.0]);
    const scale = 0.1;
    const zeroPoint = 0;

    const qOut = pool.quantizeFloat32ToInt8Fast(input, scale, zeroPoint);

    expect(qOut[0]).toBe(0);   // 0 / 0.1 = 0
    expect(qOut[1]).toBe(12);  // 1.2 / 0.1 = 12
    expect(qOut[2]).toBe(-12); // -1.2 / 0.1 = -12
    expect(qOut[3]).toBe(50);  // 5.0 / 0.1 = 50
    expect(qOut[4]).toBe(-100);// -10.0 / 0.1 = -100

    const dqOut = pool.dequantizeInt8ToFloat32Fast(qOut, scale, zeroPoint);
    expect(dqOut[0]).toBeCloseTo(0.0);
    expect(dqOut[1]).toBeCloseTo(1.2);
    expect(dqOut[2]).toBeCloseTo(-1.2);
    expect(dqOut[3]).toBeCloseTo(5.0);
    expect(dqOut[4]).toBeCloseTo(-10.0);
  });

  it('should pass zero-allocation benchmark with per-keystroke inference latency < 5ms', () => {
    const benchPool = new MemoryPool({
      int8Capacity: 65536,
      float32Capacity: 65536,
      enableSimd: true,
    });

    const result = benchPool.runZeroAllocationBenchmark({
      iterations: 1000,
      tensorSize: 256,
      targetLatencyMs: 5.0,
    });

    expect(result.iterations).toBe(1000);
    expect(result.allocationsCount).toBe(0);
    expect(result.avgLatencyMs).toBeLessThan(5.0);
    expect(result.p95LatencyMs).toBeLessThan(5.0);
    expect(result.underLatencyTarget).toBe(true);
  });
});
