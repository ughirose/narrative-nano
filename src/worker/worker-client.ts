import type { InferenceTask, InferenceResult, StateDeltaEvent, RingBufferStats } from '../types.js';
import { SPSCRingBuffer } from '../spsc/ring-buffer.js';

export interface WorkerClientOptions {
  workerScriptUrl?: string;
  workerInstance?: Worker;
  ringBufferCapacity?: number;
  ringBufferSlotSize?: number;
  enableSimd?: boolean;
}

export class InferenceWorkerClient {
  private worker: Worker | null = null;
  private inputRingBuffer: SPSCRingBuffer;
  private outputRingBuffer: SPSCRingBuffer;
  private pendingTasks = new Map<string, {
    resolve: (result: InferenceResult) => void;
    reject: (err: Error) => void;
    timeoutId: ReturnType<typeof setTimeout>;
  }>();

  private isReady = false;

  constructor(options: WorkerClientOptions = {}) {
    const capacity = options.ringBufferCapacity ?? 64;
    const slotSize = options.ringBufferSlotSize ?? 2048;

    this.inputRingBuffer = new SPSCRingBuffer({ capacity, elementSizeBytes: slotSize });
    this.outputRingBuffer = new SPSCRingBuffer({ capacity, elementSizeBytes: slotSize });

    if (options.workerInstance) {
      this.worker = options.workerInstance;
      this.attachWorkerListeners();
    } else if (options.workerScriptUrl) {
      this.worker = new Worker(options.workerScriptUrl, { type: 'module' });
      this.attachWorkerListeners();
    }
  }

  /**
   * Initializes worker and passes SPSC ring buffer SharedArrayBuffers.
   */
  async initialize(modelBuffer?: ArrayBuffer): Promise<void> {
    if (!this.worker) return;

    return new Promise((resolve) => {
      const workerRef = this.worker;
      if (!workerRef) {
        resolve();
        return;
      }

      const onInitComplete = (e: MessageEvent) => {
        if (e.data?.type === 'INIT_COMPLETE') {
          this.isReady = true;
          workerRef.removeEventListener('message', onInitComplete);
          resolve();
        }
      };

      workerRef.addEventListener('message', onInitComplete);

      workerRef.postMessage({
        type: 'INIT',
        modelBuffer,
        inputRingBuffer: this.inputRingBuffer.getBuffer(),
        outputRingBuffer: this.outputRingBuffer.getBuffer(),
        enableSimd: true,
      });
    });
  }

  private attachWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data) return;

      if (data.type === 'RESULT' && data.result) {
        this.handleResult(data.result as InferenceResult);
      }
    };
  }

  private handleResult(result: InferenceResult): void {
    const pending = this.pendingTasks.get(result.id);
    if (pending) {
      clearTimeout(pending.timeoutId);
      this.pendingTasks.delete(result.id);
      pending.resolve(result);
    }
  }

  /**
   * Evaluates a StateDeltaEvent asynchronously via Wasm SIMD inference worker.
   * Leverages SPSC Ring Buffer when available, falling back to postMessage.
   */
  async evaluateDelta(delta: StateDeltaEvent, timeoutMs: number = 5000): Promise<InferenceResult> {
    const taskId = delta.id || `task-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
    const task: InferenceTask = {
      id: taskId,
      timestamp: Date.now(),
      deltaEvent: delta,
    };

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingTasks.delete(taskId);
        reject(new Error(`Inference task ${taskId} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingTasks.set(taskId, { resolve, reject, timeoutId });

      // Try pushing to input SPSC ring buffer for zero-copy delivery
      const pushed = this.inputRingBuffer.pushObject(task);
      if (pushed && this.worker) {
        this.worker.postMessage({ type: 'PROCESS_QUEUE' });
      } else if (this.worker) {
        // Fallback to direct postMessage
        this.worker.postMessage({
          type: 'EVAL_DELTA',
          taskId,
          delta,
        });
      } else {
        // Local synchronous processing fallback (e.g. node environment or no worker)
        clearTimeout(timeoutId);
        this.pendingTasks.delete(taskId);
        const localResult: InferenceResult = {
          id: taskId,
          timestamp: Date.now(),
          accepted: delta.operation !== 'delete',
          executionTimeMs: 0.2,
          metadata: { mode: 'sync-fallback' },
        };
        resolve(localResult);
      }
    });
  }

  /**
   * Drains output ring buffer for completed results.
   */
  pollResults(): InferenceResult[] {
    const results: InferenceResult[] = [];
    while (!this.outputRingBuffer.isEmpty()) {
      const result = this.outputRingBuffer.popObject<InferenceResult>();
      if (!result) break;
      results.push(result);
      this.handleResult(result);
    }
    return results;
  }

  getInputRingBufferStats(): RingBufferStats {
    return this.inputRingBuffer.getStats();
  }

  getOutputRingBufferStats(): RingBufferStats {
    return this.outputRingBuffer.getStats();
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pendingTasks.forEach(({ reject, timeoutId }) => {
      clearTimeout(timeoutId);
      reject(new Error('Inference worker terminated'));
    });
    this.pendingTasks.clear();
    this.isReady = false;
  }

  ready(): boolean {
    return this.isReady;
  }
}
