import type { StateDeltaEvent } from '@schema';

export type { StateDeltaEvent } from '@schema';

export interface QuantizationParam {
  scale: number;
  zeroPoint: number;
}

export interface ModelLoaderOptions {
  executionProviders?: string[];
  numThreads?: number;
  enableSimd?: boolean;
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
}

export interface InferenceTask {
  id: string;
  timestamp: number;
  deltaEvent?: StateDeltaEvent;
  inputTokens?: Int32Array | number[];
  inputData?: Float32Array | Int8Array;
  metadata?: Record<string, unknown>;
}

export interface InferenceResult {
  id: string;
  timestamp: number;
  accepted: boolean;
  score?: number;
  outputLogits?: Float32Array;
  processedTokens?: number;
  executionTimeMs: number;
  metadata?: Record<string, unknown>;
}

export interface RingBufferStats {
  capacity: number;
  length: number;
  freeSlots: number;
  droppedCount: number;
  throughputOpsPerSec: number;
}

export interface PlotailorIDEPanel {
  id: string;
  title: string;
  position: 'left' | 'center' | 'right' | 'bottom';
  updateContent(htmlOrText: string): void;
}

export interface PlotailorIDE {
  registerPanel(panel: PlotailorIDEPanel): void;
  updateStatus(statusText: string): void;
  getPanels(): PlotailorIDEPanel[];
  activePane: 'left' | 'center' | 'right';
}
