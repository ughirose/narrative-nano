/**
 * Narrative-Nano WebAssembly SIMD Inference Core
 * Emscripten-compatible runtime wrapper
 */

const fs = (typeof process !== 'undefined' && process.versions && process.versions.node) ? require('fs') : null;
const path = (typeof process !== 'undefined' && process.versions && process.versions.node) ? require('path') : null;

function createNarrativeCoreModule(moduleArg = {}) {
  const Module = Object.assign({}, moduleArg);

  Module.ready = new Promise((resolve, reject) => {
    let wasmBinary = Module.wasmBinary;

    const instantiate = (bytes) => {
      const importObject = {
        env: {
          memory: Module.wasmMemory,
          abort: (msg) => { console.error('Wasm aborted:', msg); throw new Error(msg); }
        }
      };

      WebAssembly.instantiate(bytes, importObject)
        .then(({ instance, module }) => {
          Module.instance = instance;
          Module.module = module;
          Module.asm = instance.exports;

          // Wire up exports
          Module.memory = instance.exports.memory;
          Module.HEAPU8 = new Uint8Array(Module.memory.buffer);
          Module.HEAPF32 = new Float32Array(Module.memory.buffer);
          Module._init_model = instance.exports._init_model;
          Module._forward_step = instance.exports._forward_step;
          Module._free_model = instance.exports._free_model;
          Module._malloc = instance.exports._malloc || instance.exports.malloc;
          Module._free = instance.exports._free || instance.exports.free;

          // Update buffer views on memory growth
          Module.updateMemoryViews = () => {
            Module.HEAPU8 = new Uint8Array(Module.memory.buffer);
            Module.HEAPF32 = new Float32Array(Module.memory.buffer);
          };

          if (Module.onRuntimeInitialized) {
            Module.onRuntimeInitialized();
          }
          resolve(Module);
        })
        .catch(reject);
    };

    if (wasmBinary) {
      instantiate(wasmBinary);
    } else if (fs && path) {
      try {
        const wasmPath = Module.locateFile ? Module.locateFile('narrative_core.wasm') : path.join(__dirname, 'narrative_core.wasm');
        const buffer = fs.readFileSync(wasmPath);
        instantiate(buffer);
      } catch (err) {
        reject(err);
      }
    } else if (typeof fetch === 'function') {
      const wasmPath = Module.locateFile ? Module.locateFile('narrative_core.wasm') : 'narrative_core.wasm';
      fetch(wasmPath)
        .then(res => res.arrayBuffer())
        .then(instantiate)
        .catch(reject);
    } else {
      reject(new Error('Unable to find or load narrative_core.wasm'));
    }
  });

  return Module;
}

if (typeof exports === 'object' && typeof module === 'object') {
  module.exports = createNarrativeCoreModule;
  module.exports.default = createNarrativeCoreModule;
} else if (typeof define === 'function' && define.amd) {
  define([], () => createNarrativeCoreModule);
} else if (typeof window !== 'undefined') {
  window.createNarrativeCoreModule = createNarrativeCoreModule;
}
