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
export type NarrativeConnective = 'none' | 'causal' | 'adversative' | 'temporal' | 'additive';

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
  causalConnective?: NarrativeConnective;
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

export const NARRATIVE_NANO_MODEL_URL = '/models/narrative_nano_ultra_v15_qat_int8.onnx';
export const NARRATIVE_NANO_LATEST_MODEL_URL = '/models/narrative_nano_latest.onnx';
export const NARRATIVE_NANO_MODEL_NAME = 'Narrative-Nano Ultra v15 QAT INT8';
export const NARRATIVE_NANO_MODEL_PARAMS = '14.56M';
export const NARRATIVE_NANO_MODEL_HEADS = 8;

