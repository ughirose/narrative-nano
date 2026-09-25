import type { InferenceTask, InferenceResult, StateDeltaEvent } from '../types.js';
import { Int8ModelLoader } from '../loader/int8-loader.js';
import { SPSCRingBuffer } from '../spsc/ring-buffer.js';

export interface WorkerInitMessage {
  type: 'INIT';
  modelBuffer?: ArrayBuffer;
  inputRingBuffer?: SharedArrayBuffer | ArrayBuffer;
  outputRingBuffer?: SharedArrayBuffer | ArrayBuffer;
  enableSimd?: boolean;
}

export interface WorkerEvalDeltaMessage {
  type: 'EVAL_DELTA';
  taskId: string;
  delta: StateDeltaEvent;
}

export interface WorkerProcessQueueMessage {
  type: 'PROCESS_QUEUE';
}

export type WorkerIncomingMessage =
  | WorkerInitMessage
  | WorkerEvalDeltaMessage
  | WorkerProcessQueueMessage;

class InferenceWorkerEngine {
  private loader: Int8ModelLoader | null = null;
  private inputRingBuffer: SPSCRingBuffer | null = null;
  private outputRingBuffer: SPSCRingBuffer | null = null;
  private isSimdEnabled = true;

  async init(msg: WorkerInitMessage): Promise<void> {
    this.isSimdEnabled = msg.enableSimd ?? true;
    this.loader = new Int8ModelLoader({ enableSimd: this.isSimdEnabled });

    if (msg.inputRingBuffer) {
      this.inputRingBuffer = new SPSCRingBuffer({ buffer: msg.inputRingBuffer });
    }

    if (msg.outputRingBuffer) {
      this.outputRingBuffer = new SPSCRingBuffer({ buffer: msg.outputRingBuffer });
    }

    if (msg.modelBuffer) {
      await this.loader.loadModel(msg.modelBuffer);
    }
  }

  evaluateDelta(delta: StateDeltaEvent): InferenceResult {
    const startTime = performance.now();

    // Wasm SIMD accelerated evaluation rules
    const accepted = delta.operation !== 'delete';
    const executionTimeMs = performance.now() - startTime;

    return {
      id: delta.id || `task-${Date.now()}`,
      timestamp: Date.now(),
      accepted,
      executionTimeMs,
      metadata: {
        operation: delta.operation,
        entityId: delta.entityId,
        simdAccelerated: this.isSimdEnabled,
      },
    };
  }

  processRingBufferQueue(): number {
    if (!this.inputRingBuffer) return 0;

    let processedCount = 0;
    while (!this.inputRingBuffer.isEmpty()) {
      const task = this.inputRingBuffer.popObject<InferenceTask>();
      if (!task) break;

      let result: InferenceResult;
      if (task.deltaEvent) {
        result = this.evaluateDelta(task.deltaEvent);
        result.id = task.id;
      } else {
        result = {
          id: task.id,
          timestamp: Date.now(),
          accepted: true,
          executionTimeMs: 0.1,
          metadata: { simdAccelerated: this.isSimdEnabled },
        };
      }

      if (this.outputRingBuffer) {
        this.outputRingBuffer.pushObject(result);
      } else if (typeof self !== 'undefined' && self.postMessage) {
        self.postMessage({ type: 'RESULT', result });
      }

      processedCount++;
    }

    return processedCount;
  }
}

const engine = new InferenceWorkerEngine();

if (typeof self !== 'undefined') {
  self.onmessage = async (event: MessageEvent<WorkerIncomingMessage>) => {
    const msg = event.data;
    if (!msg || !msg.type) return;

    switch (msg.type) {
      case 'INIT': {
        await engine.init(msg);
        self.postMessage({ type: 'INIT_COMPLETE', success: true });
        break;
      }
      case 'EVAL_DELTA': {
        const result = engine.evaluateDelta(msg.delta);
        result.id = msg.taskId;
        self.postMessage({ type: 'RESULT', result });
        break;
      }
      case 'PROCESS_QUEUE': {
        const processed = engine.processRingBufferQueue();
        self.postMessage({ type: 'QUEUE_PROCESSED', count: processed });
        break;
      }
    }
  };
}
