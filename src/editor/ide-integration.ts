import type { PlotailorIDE, PlotailorIDEPanel, RingBufferStats } from '../types.js';
import type { MemoryPoolStats } from '../runtime/MemoryPool.js';

export class NanoIDEIntegration {
  private ide: PlotailorIDE | null = null;
  private panel: PlotailorIDEPanel | null = null;

  private simdActive = true;
  private modelLoaded = false;
  private modelName = 'ONNX INT8 Quantized Model';
  private totalEvaluations = 0;
  private ringBufferStats: RingBufferStats = {
    capacity: 64,
    length: 0,
    freeSlots: 64,
    droppedCount: 0,
    throughputOpsPerSec: 0,
  };
  private memoryPoolStats: MemoryPoolStats | null = null;

  attachEditor(ide: PlotailorIDE): void {
    this.ide = ide;

    this.panel = {
      id: 'narrative-nano-inspector',
      title: 'Narrative Nano (Wasm SIMD)',
      position: 'right', // Integrated into 3-pane right inspector panel
      updateContent: (content: string) => {
        if (typeof console !== 'undefined') {
          console.log(`[Narrative-Nano Inspector] ${content}`);
        }
      },
    };

    if (this.ide && typeof this.ide.registerPanel === 'function') {
      this.ide.registerPanel(this.panel);
    }

    this.renderInlinePanel();
  }

  updateMetrics(stats: {
    simdActive?: boolean;
    modelLoaded?: boolean;
    modelName?: string;
    totalEvaluations?: number;
    ringBufferStats?: RingBufferStats;
    memoryPoolStats?: MemoryPoolStats;
  }): void {
    if (stats.simdActive !== undefined) this.simdActive = stats.simdActive;
    if (stats.modelLoaded !== undefined) this.modelLoaded = stats.modelLoaded;
    if (stats.modelName !== undefined) this.modelName = stats.modelName;
    if (stats.totalEvaluations !== undefined) this.totalEvaluations = stats.totalEvaluations;
    if (stats.ringBufferStats !== undefined) this.ringBufferStats = stats.ringBufferStats;
    if (stats.memoryPoolStats !== undefined) this.memoryPoolStats = stats.memoryPoolStats;

    this.renderInlinePanel();
  }

  private renderInlinePanel(): void {
    const poolHtml = this.memoryPoolStats
      ? `
        <div class="memory-pool-metrics">
          <p><strong>Memory Pool Allocator:</strong> Active (Zero-Allocation)</p>
          <p><strong>Allocated Bytes:</strong> ${this.memoryPoolStats.totalAllocatedBytes} / ${this.memoryPoolStats.totalCapacityBytes}</p>
          <p><strong>Reset Count:</strong> ${this.memoryPoolStats.resetCount}</p>
        </div>
      `.trim()
      : '';

    const html = `
      <div class="nano-inspector-panel">
        <h4>Narrative Nano SIMD Inference Engine</h4>
        <p><strong>Status:</strong> ${this.modelLoaded ? 'Active (Loaded)' : 'Idle / Ready'}</p>
        <p><strong>Model:</strong> ${this.modelName}</p>
        <p><strong>SIMD 128-bit Vectorization:</strong> ${this.simdActive ? 'Enabled' : 'Disabled'}</p>
        <p><strong>Evaluations:</strong> ${this.totalEvaluations}</p>
        <div class="ring-buffer-metrics">
          <p><strong>SPSC Ring Buffer Capacity:</strong> ${this.ringBufferStats.capacity}</p>
          <p><strong>Active Queue Length:</strong> ${this.ringBufferStats.length}</p>
          <p><strong>Dropped Operations:</strong> ${this.ringBufferStats.droppedCount}</p>
        </div>
        ${poolHtml}
      </div>
    `.trim();

    if (this.panel) {
      this.panel.updateContent(html);
    }

    if (this.ide && typeof this.ide.updateStatus === 'function') {
      this.ide.updateStatus(`Nano Engine: ${this.modelLoaded ? 'Ready' : 'Initialized'} (SIMD: ${this.simdActive ? 'ON' : 'OFF'})`);
    }
  }

  getPanelContent(): string {
    return `Model: ${this.modelName}, SIMD: ${this.simdActive}, Loaded: ${this.modelLoaded}`;
  }
}
