import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  WatchdogEngine,
  CONTROL_BLOCK_INDICES,
  WatchdogStatus,
} from '../worker/WatchdogEngine';

/**
 * Mock Worker implementation for Vitest environment
 */
class MockWorker implements Worker {
  public onmessage: ((this: Worker, ev: MessageEvent) => any) | null = null;
  public onerror: ((this: AbstractWorker, ev: ErrorEvent) => any) | null = null;
  public onmessageerror: ((this: Worker, ev: MessageEvent) => any) | null = null;

  public terminated = false;
  public postedMessages: unknown[] = [];

  public postMessage(message: any, _options?: any): void {
    this.postedMessages.push(message);
  }

  public terminate(): void {
    this.terminated = true;
  }

  public addEventListener(): void {}
  public removeEventListener(): void {}
  public dispatchEvent(): boolean {
    return true;
  }

  // Test helper to simulate worker response
  public simulateResponse(data: unknown): void {
    if (this.onmessage) {
      this.onmessage.call(this as unknown as Worker, { data } as MessageEvent);
    }
  }

  // Test helper to simulate worker error
  public simulateError(message: string): void {
    if (this.onerror) {
      this.onerror.call(this as unknown as AbstractWorker, { message } as ErrorEvent);
    }
  }
}

describe('WatchdogEngine & Cold Resync Agent', () => {
  let createdWorkers: MockWorker[] = [];

  const createMockWorkerFactory = () => {
    return () => {
      const worker = new MockWorker();
      createdWorkers.push(worker);
      return worker as unknown as Worker;
    };
  };

  beforeEach(() => {
    createdWorkers = [];
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should process fast inference (<50ms) successfully', async () => {
    const factory = createMockWorkerFactory();
    const statusHistory: WatchdogStatus[] = [];

    const engine = new WatchdogEngine<string, { result: string }>({
      workerFactory: factory,
      timeoutMs: 50,
      onStatusChange: (status) => statusHistory.push(status),
    });

    expect(engine.getStatus()).toBe('idle');
    expect(createdWorkers.length).toBe(1);

    const dispatchPromise = engine.dispatch('test input');
    expect(engine.getStatus()).toBe('processing');

    const worker = createdWorkers[0];
    expect(worker.postedMessages.length).toBe(1);
    expect(worker.postedMessages[0]).toBe('test input');

    // Fast response after 10ms
    vi.advanceTimersByTime(10);
    worker.simulateResponse({ result: 'output' });

    const result = await dispatchPromise;
    expect(result).toEqual({ result: 'output' });
    expect(engine.getStatus()).toBe('idle');

    const metrics = engine.getMetrics();
    expect(metrics.totalRequests).toBe(1);
    expect(metrics.totalTimeouts).toBe(0);
    expect(metrics.resyncCount).toBe(0);
  });

  it('should trigger forced terminate() and Cold Resync when inference exceeds 50ms', async () => {
    const factory = createMockWorkerFactory();
    const onTimeout = vi.fn();
    const onResync = vi.fn();
    const onWorkerRestarted = vi.fn();

    const engine = new WatchdogEngine<string, { result: string }>({
      workerFactory: factory,
      timeoutMs: 50,
      checkIntervalMs: 5,
      onTimeout,
      onResync,
      onWorkerRestarted,
    });

    const initialWorker = createdWorkers[0];

    // Dispatch request
    const dispatchPromise = engine.dispatch('heavy request');

    // Advance time past 50ms threshold
    vi.advanceTimersByTime(55);

    // Initial worker should be terminated
    expect(initialWorker.terminated).toBe(true);
    expect(createdWorkers.length).toBe(2); // Fresh worker created

    expect(onTimeout).toHaveBeenCalledWith(expect.any(Number), 'heavy request');
    expect(onWorkerRestarted).toHaveBeenCalled();
    expect(onResync).toHaveBeenCalledWith('heavy request');

    const newWorker = createdWorkers[1];
    expect(newWorker.postedMessages.length).toBe(1);
    expect(newWorker.postedMessages[0]).toBe('heavy request');

    // Complete the resynced request on the new worker
    newWorker.simulateResponse({ result: 'recovered' });

    const result = await dispatchPromise;
    expect(result).toEqual({ result: 'recovered' });

    const metrics = engine.getMetrics();
    expect(metrics.totalTimeouts).toBe(1);
    expect(metrics.resyncCount).toBe(1);
  });

  it('should initialize and reset SharedArrayBuffer atomic pointers on Cold Resync', () => {
    const factory = createMockWorkerFactory();
    const sharedBuffer = new SharedArrayBuffer(1024);
    const controlView = new Int32Array(sharedBuffer, 0, 16);

    // Corrupt pointer state
    controlView[CONTROL_BLOCK_INDICES.REQ_HEAD] = 42;
    controlView[CONTROL_BLOCK_INDICES.REQ_TAIL] = 20;
    controlView[CONTROL_BLOCK_INDICES.RESP_HEAD] = 15;
    controlView[CONTROL_BLOCK_INDICES.RESP_TAIL] = 10;
    controlView[CONTROL_BLOCK_INDICES.WORKER_STATUS] = 1;

    const engine = new WatchdogEngine({
      workerFactory: factory,
      sharedBuffer,
    });

    // Constructor should reset pointers
    expect(controlView[CONTROL_BLOCK_INDICES.REQ_HEAD]).toBe(0);
    expect(controlView[CONTROL_BLOCK_INDICES.REQ_TAIL]).toBe(0);
    expect(controlView[CONTROL_BLOCK_INDICES.RESP_HEAD]).toBe(0);
    expect(controlView[CONTROL_BLOCK_INDICES.RESP_TAIL]).toBe(0);
    expect(controlView[CONTROL_BLOCK_INDICES.WORKER_STATUS]).toBe(0);

    // Corrupt state again
    controlView[CONTROL_BLOCK_INDICES.REQ_HEAD] = 99;
    controlView[CONTROL_BLOCK_INDICES.WORKER_STATUS] = 1;

    // Trigger explicit pointer reset
    engine.resetControlPointers();

    expect(controlView[CONTROL_BLOCK_INDICES.REQ_HEAD]).toBe(0);
    expect(controlView[CONTROL_BLOCK_INDICES.WORKER_STATUS]).toBe(0);
  });

  it('should support custom timeout threshold and respect 3-pane IDE non-modal constitution', async () => {
    const factory = createMockWorkerFactory();
    const statusLog: { status: WatchdogStatus; detail?: string }[] = [];

    const engine = new WatchdogEngine({
      workerFactory: factory,
      timeoutMs: 20, // 20ms threshold
      onStatusChange: (status, detail) => statusLog.push({ status, detail }),
    });

    engine.dispatch('fast timeout request');

    // Advance 25ms (exceeds 20ms custom threshold)
    vi.advanceTimersByTime(25);

    expect(createdWorkers[0].terminated).toBe(true);
    expect(createdWorkers.length).toBe(2);

    // Verify status transition without modal side-effects
    const resyncStatus = statusLog.find((s) => s.status === 'resyncing');
    expect(resyncStatus).toBeDefined();
    expect(resyncStatus?.detail).toContain('20ms');
  });

  it('should handle engine disposal cleanly', async () => {
    const factory = createMockWorkerFactory();
    const engine = new WatchdogEngine({
      workerFactory: factory,
    });

    const worker = createdWorkers[0];
    engine.dispose();

    expect(worker.terminated).toBe(true);
    expect(engine.getStatus()).toBe('disposed');

    await expect(engine.dispatch('data')).rejects.toThrow('disposed');
  });
});
