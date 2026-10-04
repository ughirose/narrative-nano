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

export type NarrativeModality = 'dialogue' | 'narration';
export type NarrativeEventAction = 'none' | 'acquire' | 'drop' | 'move' | 'speak' | 'state_change';

export interface EntitySpan {
  start: number;
  end: number;
  text?: string;
  category: 'character' | 'item' | 'location' | 'concept';
  confidence: number;
}

export interface InferenceResult {
  id: string;
  timestamp: number;
  accepted: boolean;
  score?: number;
  modality?: NarrativeModality;
  eventAction?: NarrativeEventAction;
  epistemicScore?: number;
  entitySpans?: EntitySpan[];
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
