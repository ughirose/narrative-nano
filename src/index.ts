import type { StateDeltaEvent } from '@schema';
import type { PlotailorIDE } from '@editor';

export * from './worker/WatchdogEngine';

export class NanoInferenceWasm {
  evaluateDelta(event: StateDeltaEvent): boolean {
    return event.operation !== 'delete';
  }

  attachEditor(editor: PlotailorIDE): void {
    console.log('Attached to editor', editor);
  }
}
