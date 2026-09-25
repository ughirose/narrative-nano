/**
 * [TASK-318] 50ms Exceeded Decision Watchdog & Self-Healing Cold Resync Agent
 *
 * Target Repository: ughirose/narrative-nano
 * Module: WatchdogEngine.ts
 *
 * Spec Compliance:
 * - 50ms max response watchdog timer with Atomics API & millisecond timer integration
 * - Forced Worker terminate() and immediate relaunch on >50ms timeout
 * - SPSC Ring Buffer atomic pointer initialization and safe Cold Resync of recent request
 * - 3-pane integrated IDE compliance (no modals, inline state observables)
 */

export const CONTROL_BLOCK_INDICES = {
  REQ_HEAD: 0,
  REQ_TAIL: 1,
  RESP_HEAD: 2,
  RESP_TAIL: 3,
  WORKER_STATUS: 4, // 0: IDLE, 1: PROCESSING, 2: TIMEOUT_ERROR
  HEARTBEAT: 5,
} as const;

export type WatchdogStatus = 'idle' | 'processing' | 'resyncing' | 'error' | 'disposed';

export interface WatchdogMetrics {
  totalRequests: number;
  totalTimeouts: number;
  resyncCount: number;
  lastLatencyMs: number;
  avgLatencyMs: number;
}

export interface WatchdogOptions<TReq = unknown, TRes = unknown> {
  /** Factory function to create a new Worker instance */
  workerFactory: () => Worker;
  /** Timeout threshold in milliseconds. Default: 50ms */
  timeoutMs?: number;
  /** Timer check interval in milliseconds. Default: 5ms */
  checkIntervalMs?: number;
  /** Optional SharedArrayBuffer for SPSC Ring Buffer IPC */
  sharedBuffer?: SharedArrayBuffer;
  /** Callback triggered when worker times out (>50ms) */
  onTimeout?: (elapsedMs: number, lastRequest: TReq | null) => void;
  /** Callback triggered when Cold Resync is performed */
  onResync?: (lastRequest: TReq | null) => void;
  /** Callback triggered when worker is recreated */
  onWorkerRestarted?: (newWorker: Worker) => void;
  /** Non-modal status observer for 3-pane integrated IDE */
  onStatusChange?: (status: WatchdogStatus, detail?: string) => void;
  /** Callback for inference response */
  onResponse?: (response: TRes, latencyMs: number) => void;
}

export class WatchdogEngine<TReq = unknown, TRes = unknown> {
  private worker: Worker;
  private readonly timeoutMs: number;
  private readonly checkIntervalMs: number;
  private readonly sharedBuffer?: SharedArrayBuffer;
  private readonly controlView?: Int32Array;

  private status: WatchdogStatus = 'idle';
  private timerId: ReturnType<typeof setInterval> | null = null;
  private currentRequestStartMs: number | null = null;
  private lastRequest: TReq | null = null;

  private activeResolver: {
    resolve: (val: TRes) => void;
    reject: (reason: unknown) => void;
  } | null = null;

  private metrics: WatchdogMetrics = {
    totalRequests: 0,
    totalTimeouts: 0,
    resyncCount: 0,
    lastLatencyMs: 0,
    avgLatencyMs: 0,
  };

  constructor(private readonly options: WatchdogOptions<TReq, TRes>) {
    this.timeoutMs = options.timeoutMs ?? 50;
    this.checkIntervalMs = options.checkIntervalMs ?? 5;
    this.sharedBuffer = options.sharedBuffer;

    if (this.sharedBuffer) {
      this.controlView = new Int32Array(this.sharedBuffer, 0, 16);
      this.resetControlPointers();
    }

    this.worker = this.options.workerFactory();
    this.setupWorkerHandlers(this.worker);
    this.setStatus('idle');
  }

  /**
   * Initialize or reset SPSC Ring Buffer atomic pointers in SharedArrayBuffer
   */
  public resetControlPointers(): void {
    if (!this.controlView) return;
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.REQ_HEAD, 0);
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.REQ_TAIL, 0);
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.RESP_HEAD, 0);
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.RESP_TAIL, 0);
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.WORKER_STATUS, 0);
    Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.HEARTBEAT, Date.now() % 1000000);
  }

  /**
   * Get current non-modal watchdog status
   */
  public getStatus(): WatchdogStatus {
    return this.status;
  }

  /**
   * Get performance metrics
   */
  public getMetrics(): Readonly<WatchdogMetrics> {
    return { ...this.metrics };
  }

  /**
   * Get current worker instance
   */
  public getWorker(): Worker {
    return this.worker;
  }

  /**
   * Dispatch request to worker with 50ms watchdog monitoring
   */
  public async dispatch(requestData: TReq): Promise<TRes> {
    if (this.status === 'disposed') {
      throw new Error('WatchdogEngine has been disposed.');
    }

    this.lastRequest = requestData;
    this.metrics.totalRequests++;
    this.currentRequestStartMs = performance.now();
    this.setStatus('processing');

    if (this.controlView) {
      Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.WORKER_STATUS, 1);
      Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.HEARTBEAT, Math.floor(this.currentRequestStartMs));
    }

    this.startWatchdogTimer();

    return new Promise<TRes>((resolve, reject) => {
      this.activeResolver = { resolve, reject };

      try {
        if (requestData instanceof ArrayBuffer) {
          this.worker.postMessage(requestData, [requestData]);
        } else {
          this.worker.postMessage(requestData);
        }
      } catch (err) {
        this.stopWatchdogTimer();
        this.setStatus('error', String(err));
        reject(err);
      }
    });
  }

  /**
   * Handle incoming message from worker
   */
  private handleWorkerMessage(event: MessageEvent): void {
    if (this.status === 'disposed') return;

    const endMs = performance.now();
    const latencyMs = this.currentRequestStartMs ? endMs - this.currentRequestStartMs : 0;

    this.stopWatchdogTimer();
    this.updateLatencyMetrics(latencyMs);

    if (this.controlView) {
      Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.WORKER_STATUS, 0);
    }

    this.setStatus('idle');

    const response = event.data as TRes;
    if (this.options.onResponse) {
      this.options.onResponse(response, latencyMs);
    }

    if (this.activeResolver) {
      const resolver = this.activeResolver;
      this.activeResolver = null;
      resolver.resolve(response);
    }
  }

  /**
   * Handle worker error
   */
  private handleWorkerError(error: ErrorEvent | unknown): void {
    if (this.status === 'disposed') return;
    this.stopWatchdogTimer();
    this.setStatus('error', error instanceof ErrorEvent ? error.message : String(error));

    if (this.activeResolver) {
      const resolver = this.activeResolver;
      this.activeResolver = null;
      resolver.reject(error);
    }
  }

  /**
   * Start watchdog interval checking for >50ms timeout
   */
  private startWatchdogTimer(): void {
    this.stopWatchdogTimer();

    this.timerId = setInterval(() => {
      this.checkTimeout();
    }, this.checkIntervalMs);
  }

  /**
   * Stop watchdog interval timer
   */
  private stopWatchdogTimer(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  /**
   * Check if active request has exceeded threshold (50ms)
   */
  public checkTimeout(): void {
    if (this.status !== 'processing' || this.currentRequestStartMs === null) {
      return;
    }

    const elapsed = performance.now() - this.currentRequestStartMs;

    // Direct millisecond timer or Atomics API check
    if (elapsed >= this.timeoutMs) {
      this.performColdResync(elapsed);
    }
  }

  /**
   * Perform Cold Resync: Terminate hung worker, reset SAB pointers, launch new Worker, re-send request
   */
  public performColdResync(elapsedMs: number): void {
    this.stopWatchdogTimer();
    this.metrics.totalTimeouts++;
    this.metrics.resyncCount++;

    if (this.controlView) {
      Atomics.store(this.controlView, CONTROL_BLOCK_INDICES.WORKER_STATUS, 2);
    }

    this.setStatus('resyncing', `Timeout exceeded (${elapsedMs.toFixed(1)}ms > ${this.timeoutMs}ms)`);

    if (this.options.onTimeout) {
      this.options.onTimeout(elapsedMs, this.lastRequest);
    }

    // Force terminate hung Worker
    try {
      this.worker.terminate();
    } catch {
      // Ignore termination error if worker already stopped
    }

    // Reset Ring Buffer atomic pointers
    this.resetControlPointers();

    // Relaunch fresh Worker instance
    this.worker = this.options.workerFactory();
    this.setupWorkerHandlers(this.worker);

    if (this.options.onWorkerRestarted) {
      this.options.onWorkerRestarted(this.worker);
    }

    const resyncReq = this.lastRequest;

    if (this.options.onResync) {
      this.options.onResync(resyncReq);
    }

    // Cold Resync: safely re-dispatch recent request if available
    if (resyncReq !== null) {
      this.currentRequestStartMs = performance.now();
      this.startWatchdogTimer();
      this.setStatus('processing');

      try {
        if (resyncReq instanceof ArrayBuffer) {
          // Send copy if transferable
          const copy = resyncReq.slice(0);
          this.worker.postMessage(copy, [copy]);
        } else {
          this.worker.postMessage(resyncReq);
        }
      } catch (err) {
        this.handleWorkerError(err);
      }
    } else {
      this.setStatus('idle');
    }
  }

  private setupWorkerHandlers(worker: Worker): void {
    worker.onmessage = (e: MessageEvent) => this.handleWorkerMessage(e);
    worker.onerror = (e: ErrorEvent) => this.handleWorkerError(e);
  }

  private setStatus(newStatus: WatchdogStatus, detail?: string): void {
    this.status = newStatus;
    if (this.options.onStatusChange) {
      this.options.onStatusChange(newStatus, detail);
    }
  }

  private updateLatencyMetrics(latencyMs: number): void {
    this.metrics.lastLatencyMs = latencyMs;
    const n = this.metrics.totalRequests;
    this.metrics.avgLatencyMs = (this.metrics.avgLatencyMs * (n - 1) + latencyMs) / n;
  }

  /**
   * Clean up and dispose resources
   */
  public dispose(): void {
    this.stopWatchdogTimer();
    try {
      this.worker.terminate();
    } catch {
      // ignore
    }
    this.setStatus('disposed');
    if (this.activeResolver) {
      this.activeResolver.reject(new Error('WatchdogEngine disposed'));
      this.activeResolver = null;
    }
  }
}
