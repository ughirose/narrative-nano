import * as ort from 'onnxruntime-web';
import type { ModelLoaderOptions, QuantizationParam } from '../types.js';

export interface Int8ModelMetadata {
  inputNames: string[];
  outputNames: string[];
  inputQuantParams: Record<string, QuantizationParam>;
  outputQuantParams: Record<string, QuantizationParam>;
}

export class Int8ModelLoader {
  private session: ort.InferenceSession | null = null;
  private options: ModelLoaderOptions;
  private metadata: Int8ModelMetadata = {
    inputNames: [],
    outputNames: [],
    inputQuantParams: {},
    outputQuantParams: {},
  };

  constructor(options: ModelLoaderOptions = {}) {
    this.options = {
      numThreads: 1,
      enableSimd: true,
      graphOptimizationLevel: 'all',
      ...options,
    };

    this.configureRuntime();
  }

  /**
   * Configures global ONNX Runtime Web environment for Wasm SIMD 128-bit execution.
   */
  private configureRuntime(): void {
    if (ort.env && ort.env.wasm) {
      ort.env.wasm.simd = this.options.enableSimd ?? true;
      if (this.options.numThreads !== undefined) {
        ort.env.wasm.numThreads = this.options.numThreads;
      }
    }
  }

  /**
   * Loads an INT8 quantized ONNX model from ArrayBuffer, Uint8Array, or URL.
   */
  async loadModel(modelBufferOrUrl: ArrayBuffer | Uint8Array | string): Promise<ort.InferenceSession> {
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: this.options.executionProviders || ['wasm'],
      graphOptimizationLevel: this.options.graphOptimizationLevel || 'all',
    };

    if (typeof modelBufferOrUrl === 'string') {
      this.session = await ort.InferenceSession.create(modelBufferOrUrl, sessionOptions);
    } else {
      const buffer = modelBufferOrUrl instanceof Uint8Array
        ? modelBufferOrUrl.buffer.slice(
            modelBufferOrUrl.byteOffset,
            modelBufferOrUrl.byteOffset + modelBufferOrUrl.byteLength
          )
        : modelBufferOrUrl;
      this.session = await ort.InferenceSession.create(buffer, sessionOptions);
    }

    this.inspectMetadata();
    return this.session;
  }

  private inspectMetadata(): void {
    if (!this.session) return;
    this.metadata.inputNames = [...this.session.inputNames];
    this.metadata.outputNames = [...this.session.outputNames];
  }

  /**
   * Quantizes a Float32Array into Int8Array using scale and zeroPoint.
   * x_quant = clamp(round(x / scale) + zeroPoint, -128, 127)
   */
  static quantizeFloat32ToInt8(
    input: Float32Array,
    scale: number,
    zeroPoint: number = 0
  ): Int8Array {
    const output = new Int8Array(input.length);
    const invScale = 1.0 / scale;
    for (let i = 0; i < input.length; i++) {
      const val = Math.round(input[i] * invScale) + zeroPoint;
      output[i] = Math.max(-128, Math.min(127, val));
    }
    return output;
  }

  /**
   * Dequantizes an Int8Array back into Float32Array using scale and zeroPoint.
   * x = (x_quant - zeroPoint) * scale
   */
  static dequantizeInt8ToFloat32(
    input: Int8Array,
    scale: number,
    zeroPoint: number = 0
  ): Float32Array {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      output[i] = (input[i] - zeroPoint) * scale;
    }
    return output;
  }

  /**
   * Helper to create an INT8 ORT Tensor.
   */
  static createInt8Tensor(data: Int8Array, dims: number[]): ort.Tensor {
    return new ort.Tensor('int8', data, dims);
  }

  /**
   * Helper to create a Float32 ORT Tensor.
   */
  static createFloat32Tensor(data: Float32Array, dims: number[]): ort.Tensor {
    return new ort.Tensor('float32', data, dims);
  }

  /**
   * Executes inference with the loaded INT8 ONNX session.
   */
  async runInference(
    feeds: Record<string, ort.Tensor>
  ): Promise<Record<string, ort.Tensor>> {
    if (!this.session) {
      throw new Error('ONNX session is not initialized. Call loadModel() first.');
    }
    const results = await this.session.run(feeds);
    return results;
  }

  getMetadata(): Int8ModelMetadata {
    return { ...this.metadata };
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  async release(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}
