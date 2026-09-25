import { describe, it, expect, vi } from 'vitest';
import { NanoInferenceWasm } from '../index.js';
import type { StateDeltaEvent, PlotailorIDE, PlotailorIDEPanel } from '../types.js';

describe('NanoInferenceWasm', () => {
  it('should evaluate delta events synchronously', () => {
    const nano = new NanoInferenceWasm();

    const createDelta: StateDeltaEvent = {
      id: 'd1',
      timestamp: Date.now(),
      entityId: 'ent-1',
      operation: 'create',
    };

    const deleteDelta: StateDeltaEvent = {
      id: 'd2',
      timestamp: Date.now(),
      entityId: 'ent-1',
      operation: 'delete',
    };

    expect(nano.evaluateDelta(createDelta)).toBe(true);
    expect(nano.evaluateDelta(deleteDelta)).toBe(false);
  });

  it('should evaluate delta events asynchronously via worker client fallback', async () => {
    const nano = new NanoInferenceWasm();

    const updateDelta: StateDeltaEvent = {
      id: 'd3',
      timestamp: Date.now(),
      entityId: 'ent-2',
      operation: 'update',
    };

    const result = await nano.evaluateDeltaAsync(updateDelta);
    expect(result).toBe(true);
  });

  it('should attach to PlotailorIDE and register inspector panel', () => {
    const nano = new NanoInferenceWasm();
    let registeredPanel: PlotailorIDEPanel | null = null;
    let updatedStatus = '';

    const mockIde: PlotailorIDE = {
      activePane: 'right',
      registerPanel: (panel: PlotailorIDEPanel) => {
        registeredPanel = panel;
      },
      updateStatus: (statusText: string) => {
        updatedStatus = statusText;
      },
      getPanels: () => (registeredPanel ? [registeredPanel] : []),
    };

    nano.attachEditor(mockIde);

    expect(registeredPanel).not.toBeNull();
    if (registeredPanel) {
      const panel: PlotailorIDEPanel = registeredPanel;
      expect(panel.id).toBe('narrative-nano-inspector');
      expect(panel.position).toBe('right');
    }
    expect(updatedStatus).toContain('Nano Engine: Initialized');
  });
});
