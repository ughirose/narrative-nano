import { describe, it, expect } from 'vitest';
import { SPSCRingBuffer } from '../spsc/ring-buffer.js';

describe('SPSCRingBuffer', () => {
  it('should initialize with power-of-two capacity', () => {
    const rb = new SPSCRingBuffer({ capacity: 10, elementSizeBytes: 128 });
    expect(rb.capacity).toBe(16); // Rounded to 16
    expect(rb.elementSizeBytes).toBe(128);
    expect(rb.isEmpty()).toBe(true);
  });

  it('should push and pop raw byte payloads', () => {
    const rb = new SPSCRingBuffer({ capacity: 4, elementSizeBytes: 64 });
    const payload = new Uint8Array([10, 20, 30, 40]);

    const pushed = rb.push(payload);
    expect(pushed).toBe(true);
    expect(rb.isEmpty()).toBe(false);

    const popped = rb.pop();
    expect(popped).not.toBeNull();
    expect(popped![0]).toBe(10);
    expect(popped![1]).toBe(20);
    expect(popped![2]).toBe(30);
    expect(popped![3]).toBe(40);

    expect(rb.isEmpty()).toBe(true);
  });

  it('should push and pop JSON objects zero-copy', () => {
    const rb = new SPSCRingBuffer({ capacity: 8, elementSizeBytes: 256 });
    const task = { id: 'task-123', value: 42, text: 'narrative' };

    expect(rb.pushObject(task)).toBe(true);

    const retrieved = rb.popObject<typeof task>();
    expect(retrieved).toEqual(task);
  });

  it('should correctly null-terminate when overwriting slot with shorter payload', () => {
    const rb = new SPSCRingBuffer({ capacity: 2, elementSizeBytes: 256 });
    const longTask = { id: 'long-task-id-1234567890', description: 'a very long description string to fill up slot bytes' };
    const shortTask = { id: 's' };

    // Write long task, then pop it
    rb.pushObject(longTask);
    const poppedLong = rb.popObject<typeof longTask>();
    expect(poppedLong).toEqual(longTask);

    // Push two more items to force slot index wrap around
    rb.pushObject({ id: 'dummy1' });
    rb.popObject();

    rb.pushObject(shortTask);
    const poppedShort = rb.popObject<typeof shortTask>();
    expect(poppedShort).toEqual(shortTask);
  });

  it('should handle full buffer conditions and increment droppedCount', () => {
    const rb = new SPSCRingBuffer({ capacity: 2, elementSizeBytes: 32 });

    expect(rb.push(new Uint8Array([1]))).toBe(true);
    expect(rb.push(new Uint8Array([2]))).toBe(true);

    // Third push should fail because capacity is 2
    expect(rb.push(new Uint8Array([3]))).toBe(false);

    const stats = rb.getStats();
    expect(stats.length).toBe(2);
    expect(stats.freeSlots).toBe(0);
    expect(stats.droppedCount).toBe(1);
  });

  it('should clear buffer contents', () => {
    const rb = new SPSCRingBuffer({ capacity: 4, elementSizeBytes: 32 });
    rb.push(new Uint8Array([1, 2, 3]));
    expect(rb.isEmpty()).toBe(false);

    rb.clear();
    expect(rb.isEmpty()).toBe(true);
  });
});
