import { describe, it, expect } from 'vitest';
import { NanoInferenceWasm } from '../src/index';

describe('NanoInferenceWasm', () => {
  it('should evaluate delta correctly', () => {
    const nano = new NanoInferenceWasm();
    expect(nano.evaluateDelta({ id: '1', timestamp: 1, entityId: 'e1', operation: 'create' })).toBe(true);
    expect(nano.evaluateDelta({ id: '2', timestamp: 2, entityId: 'e2', operation: 'delete' })).toBe(false);
  });
});
