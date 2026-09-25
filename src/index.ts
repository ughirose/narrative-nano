import type { StateDeltaEvent } from '@schema';
import type { PlotailorIDE, InferenceResult } from './types.js';
import { Int8ModelLoader } from './loader/int8-loader.js';
import { SPSCRingBuffer } from './spsc/ring-buffer.js';
import { InferenceWorkerClient } from './worker/worker-client.js';
import { NanoIDEIntegration } from './editor/ide-integration.js';
import { MemoryPool, type MemoryPoolOptions, type MemoryPoolStats, type BenchmarkResult } from './runtime/MemoryPool.js';

export * from './types.js';
export { Int8ModelLoader, type Int8ModelMetadata } from './loader/int8-loader.js';
export { SPSCRingBuffer, type RingBufferOptions } from './spsc/ring-buffer.js';
export { InferenceWorkerClient, type WorkerClientOptions } from './worker/worker-client.js';
export { NanoIDEIntegration } from './editor/ide-integration.js';
export * from './model/BiaffinePASHead.js';
export { MemoryPool, type MemoryPoolOptions, type MemoryPoolStats, type BenchmarkResult } from './runtime/MemoryPool.js';

export class NanoInferenceWasm {
  private loader: Int8ModelLoader;
  private workerClient: InferenceWorkerClient;
  private ideIntegration: NanoIDEIntegration;
  private memoryPool: MemoryPool;
  private evalCount = 0;

  constructor(memoryPoolOptions?: MemoryPoolOptions) {
    this.loader = new Int8ModelLoader({ enableSimd: true });
    this.workerClient = new InferenceWorkerClient({ enableSimd: true });
    this.ideIntegration = new NanoIDEIntegration();
    this.memoryPool = new MemoryPool(memoryPoolOptions);
  }

  /**
   * Loads an INT8 quantized ONNX model into the inference engine.
   */
  async loadModel(modelBufferOrUrl: ArrayBuffer | Uint8Array | string): Promise<void> {
    await this.loader.loadModel(modelBufferOrUrl);

    let modelBuf: ArrayBuffer | undefined;
    if (modelBufferOrUrl instanceof Uint8Array) {
      modelBuf = modelBufferOrUrl.buffer.slice(
        modelBufferOrUrl.byteOffset,
        modelBufferOrUrl.byteOffset + modelBufferOrUrl.byteLength
      ) as ArrayBuffer;
    } else if (modelBufferOrUrl instanceof ArrayBuffer) {
      modelBuf = modelBufferOrUrl;
    }

    await this.workerClient.initialize(modelBuf);

    this.ideIntegration.updateMetrics({
      modelLoaded: true,
      simdActive: true,
      ringBufferStats: this.workerClient.getInputRingBufferStats(),
      memoryPoolStats: this.memoryPool.getStats(),
    });
  }

  /**
   * Synchronous evaluation of StateDeltaEvent with static memory pooling.
   */
  evaluateDelta(event: StateDeltaEvent): boolean {
    this.evalCount++;
    this.memoryPool.reset();
    const accepted = event.operation !== 'delete';

    this.ideIntegration.updateMetrics({
      totalEvaluations: this.evalCount,
      ringBufferStats: this.workerClient.getInputRingBufferStats(),
      memoryPoolStats: this.memoryPool.getStats(),
    });

    return accepted;
  }

  /**
   * Non-blocking asynchronous evaluation of StateDeltaEvent via Wasm SIMD Web Worker.
   */
  async evaluateDeltaAsync(event: StateDeltaEvent): Promise<boolean> {
    this.evalCount++;
    this.memoryPool.reset();
    const result: InferenceResult = await this.workerClient.evaluateDelta(event);

    this.ideIntegration.updateMetrics({
      totalEvaluations: this.evalCount,
      ringBufferStats: this.workerClient.getInputRingBufferStats(),
      memoryPoolStats: this.memoryPool.getStats(),
    });

    return result.accepted;
  }

  /**
   * Binds Narrative-Nano inference inspector panel to the 3-pane IDE layout.
   */
  attachEditor(editor: PlotailorIDE): void {
    this.ideIntegration.attachEditor(editor);
  }

  getLoader(): Int8ModelLoader {
    return this.loader;
  }

  getWorkerClient(): InferenceWorkerClient {
    return this.workerClient;
  }

  getMemoryPool(): MemoryPool {
    return this.memoryPool;
  }
}
