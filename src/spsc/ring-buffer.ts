import type { RingBufferStats } from '../types.js';

export interface RingBufferOptions {
  capacity?: number;          // Number of slots (must be power of 2 for fast modulo, default 64)
  elementSizeBytes?: number;  // Size of each slot in bytes (default 1024)
  buffer?: SharedArrayBuffer | ArrayBuffer;
}

const HEADER_INTS = 8;
const HEAD_INDEX = 0;
const TAIL_INDEX = 1;
const CAPACITY_INDEX = 2;
const SLOT_SIZE_INDEX = 3;
const DROPPED_INDEX = 4;

export class SPSCRingBuffer {
  private sharedBuffer: SharedArrayBuffer | ArrayBuffer;
  private header: Int32Array;
  private storage: Uint8Array;
  private isShared: boolean;

  readonly capacity: number;
  readonly elementSizeBytes: number;

  constructor(options: RingBufferOptions = {}) {
    const rawCapacity = options.capacity ?? 64;
    // Round capacity to nearest power of 2 for fast bitwise masking
    this.capacity = SPSCRingBuffer.nextPowerOfTwo(rawCapacity);
    this.elementSizeBytes = options.elementSizeBytes ?? 1024;

    const headerByteSize = HEADER_INTS * 4;
    const totalByteSize = headerByteSize + this.capacity * this.elementSizeBytes;

    if (options.buffer) {
      this.sharedBuffer = options.buffer;
      this.header = new Int32Array(this.sharedBuffer, 0, HEADER_INTS);
      this.isShared = typeof SharedArrayBuffer !== 'undefined' && this.sharedBuffer instanceof SharedArrayBuffer;
      this.capacity = this.header[CAPACITY_INDEX];
      this.elementSizeBytes = this.header[SLOT_SIZE_INDEX];
      this.storage = new Uint8Array(this.sharedBuffer, headerByteSize);
    } else {
      if (typeof SharedArrayBuffer !== 'undefined') {
        this.sharedBuffer = new SharedArrayBuffer(totalByteSize);
        this.isShared = true;
      } else {
        this.sharedBuffer = new ArrayBuffer(totalByteSize);
        this.isShared = false;
      }

      this.header = new Int32Array(this.sharedBuffer, 0, HEADER_INTS);
      this.storage = new Uint8Array(this.sharedBuffer, headerByteSize);

      // Initialize header fields
      if (this.isShared) {
        Atomics.store(this.header, HEAD_INDEX, 0);
        Atomics.store(this.header, TAIL_INDEX, 0);
        Atomics.store(this.header, CAPACITY_INDEX, this.capacity);
        Atomics.store(this.header, SLOT_SIZE_INDEX, this.elementSizeBytes);
        Atomics.store(this.header, DROPPED_INDEX, 0);
      } else {
        this.header[HEAD_INDEX] = 0;
        this.header[TAIL_INDEX] = 0;
        this.header[CAPACITY_INDEX] = this.capacity;
        this.header[SLOT_SIZE_INDEX] = this.elementSizeBytes;
        this.header[DROPPED_INDEX] = 0;
      }
    }
  }

  private static nextPowerOfTwo(n: number): number {
    let p = 1;
    while (p < n) {
      p <<= 1;
    }
    return Math.max(2, p);
  }

  getBuffer(): SharedArrayBuffer | ArrayBuffer {
    return this.sharedBuffer;
  }

  /**
   * Pushes raw payload bytes into the SPSC Ring Buffer. Zero-copy write directly into SharedArrayBuffer slot.
   */
  push(payload: Uint8Array | ArrayBufferView): boolean {
    const head = this.isShared ? Atomics.load(this.header, HEAD_INDEX) : this.header[HEAD_INDEX];
    const tail = this.isShared ? Atomics.load(this.header, TAIL_INDEX) : this.header[TAIL_INDEX];

    if (head - tail >= this.capacity) {
      // Ring buffer is full
      if (this.isShared) {
        Atomics.add(this.header, DROPPED_INDEX, 1);
      } else {
        this.header[DROPPED_INDEX]++;
      }
      return false;
    }

    const payloadBytes = new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
    const writeLen = Math.min(payloadBytes.byteLength, this.elementSizeBytes);

    const slotIndex = head & (this.capacity - 1); // Power of 2 mask
    const slotOffset = slotIndex * this.elementSizeBytes;

    // Write payload into slot
    this.storage.set(payloadBytes.subarray(0, writeLen), slotOffset);
    if (writeLen < this.elementSizeBytes) {
      this.storage[slotOffset + writeLen] = 0; // Null-terminate string/JSON payloads
    }

    // Increment head atomically
    if (this.isShared) {
      Atomics.store(this.header, HEAD_INDEX, head + 1);
      Atomics.notify(this.header, HEAD_INDEX, 1);
    } else {
      this.header[HEAD_INDEX] = head + 1;
    }

    return true;
  }

  /**
   * Pops payload bytes from the SPSC Ring Buffer into outBuffer or new Uint8Array.
   */
  pop(outBuffer?: Uint8Array): Uint8Array | null {
    const head = this.isShared ? Atomics.load(this.header, HEAD_INDEX) : this.header[HEAD_INDEX];
    const tail = this.isShared ? Atomics.load(this.header, TAIL_INDEX) : this.header[TAIL_INDEX];

    if (tail >= head) {
      // Ring buffer is empty
      return null;
    }

    const slotIndex = tail & (this.capacity - 1);
    const slotOffset = slotIndex * this.elementSizeBytes;

    const result = outBuffer && outBuffer.length >= this.elementSizeBytes
      ? outBuffer
      : new Uint8Array(this.elementSizeBytes);

    result.set(this.storage.subarray(slotOffset, slotOffset + this.elementSizeBytes));

    // Increment tail atomically
    if (this.isShared) {
      Atomics.store(this.header, TAIL_INDEX, tail + 1);
      Atomics.notify(this.header, TAIL_INDEX, 1);
    } else {
      this.header[TAIL_INDEX] = tail + 1;
    }

    return result;
  }

  /**
   * Returns current statistics of the ring buffer.
   */
  getStats(): RingBufferStats {
    const head = this.isShared ? Atomics.load(this.header, HEAD_INDEX) : this.header[HEAD_INDEX];
    const tail = this.isShared ? Atomics.load(this.header, TAIL_INDEX) : this.header[TAIL_INDEX];
    const dropped = this.isShared ? Atomics.load(this.header, DROPPED_INDEX) : this.header[DROPPED_INDEX];

    const length = Math.max(0, head - tail);
    const freeSlots = Math.max(0, this.capacity - length);

    return {
      capacity: this.capacity,
      length,
      freeSlots,
      droppedCount: dropped,
      throughputOpsPerSec: head,
    };
  }

  /**
   * Helper to serialize a JSON-encodable object into Uint8Array slot format.
   */
  pushObject<T>(obj: T): boolean {
    const str = JSON.stringify(obj);
    const textEncoder = new TextEncoder();
    const encoded = textEncoder.encode(str);
    return this.push(encoded);
  }

  /**
   * Helper to pop and parse a JSON-encodable object from slot format.
   */
  popObject<T>(): T | null {
    const raw = this.pop();
    if (!raw) return null;

    // Find null-byte or trim string length
    let end = raw.indexOf(0);
    if (end === -1) end = raw.length;

    const slice = raw.subarray(0, end);
    const textDecoder = new TextDecoder();
    const str = textDecoder.decode(slice).trim();
    if (!str) return null;

    try {
      return JSON.parse(str) as T;
    } catch {
      return null;
    }
  }

  isEmpty(): boolean {
    const head = this.isShared ? Atomics.load(this.header, HEAD_INDEX) : this.header[HEAD_INDEX];
    const tail = this.isShared ? Atomics.load(this.header, TAIL_INDEX) : this.header[TAIL_INDEX];
    return tail >= head;
  }

  clear(): void {
    if (this.isShared) {
      Atomics.store(this.header, HEAD_INDEX, 0);
      Atomics.store(this.header, TAIL_INDEX, 0);
      Atomics.store(this.header, DROPPED_INDEX, 0);
    } else {
      this.header[HEAD_INDEX] = 0;
      this.header[TAIL_INDEX] = 0;
      this.header[DROPPED_INDEX] = 0;
    }
  }
}
