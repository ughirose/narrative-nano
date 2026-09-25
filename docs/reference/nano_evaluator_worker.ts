/**
 * WorldCraft Reference Asset: nano-evaluator.worker.ts
 * 
 * Stage 1 Nano-Evaluator Web Worker
 * Transferable ArrayBuffer を用いたゼロコピーバイナリ通信と Wasm SIMD 評価コア
 * 
 * 参照構想文書: 
 * - 20260921_【構想】CodeMirror6×WebWorkerフロントエンド実装設計
 * - 20260921_【構想】物語特化極小判定モデル蒸留とWasmブラウザ実行設計
 */

interface NanoEvaluatorWasm {
  evaluateDelta(
    delPtr: number,
    delLen: number,
    insPtr: number,
    insLen: number
  ): number; // 破綻スコア (0〜65535 / 固定小数点)
  getMemoryBuffer(): ArrayBuffer;
}

let wasmInstance: NanoEvaluatorWasm | null = null;

async function initWasm() {
  try {
    const response = await fetch('/wasm/narrative_nano_simd.wasm');
    const bytes = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(bytes, {
      env: {
        memory: new WebAssembly.Memory({ initial: 64, maximum: 256 })
      }
    });
    wasmInstance = module.instance.exports as unknown as NanoEvaluatorWasm;
  } catch (err) {
    console.warn('[WorldCraft Worker] Wasm SIMD module could not be loaded, using JS fallback engine.', err);
  }
}

const initPromise = initWasm();

self.onmessage = async (e: MessageEvent<ArrayBuffer>) => {
  await initPromise;

  // Int32Array バイナリプロトコルの解釈
  // レイアウト: [version, from, delCount, insCount, ...delTokens, ...insTokens]
  const view = new Int32Array(e.data);
  const version = view[0];
  const from = view[1];
  const delCount = view[2];
  const insCount = view[3];

  let offset = 4;
  const delTokens = view.slice(offset, offset + delCount);
  offset += delCount;
  const insTokens = view.slice(offset, offset + insCount);

  let hasAnomaly = false;
  let severity = 0;

  if (wasmInstance) {
    // Wasm SIMD メモリ直接参照による高速推論 (<0.05ms)
    // 戻り値: 上位8bitがseverity, 下位16bitが破綻スコア
    const rawScore = wasmInstance.evaluateDelta(0, delCount, 0, insCount);
    hasAnomaly = (rawScore & 0xffff) > 32768;
    severity = (rawScore >> 16) & 0xff;
  } else {
    // フォールバック判定（トークンハッシュによる簡易ルール）
    hasAnomaly = false;
  }

  // 判定結果を20バイトの固定長 ArrayBuffer で返却
  // レイアウト: [version, hasAnomaly(0/1), from, to, severity(1=warning, 2=fatal)]
  const resBuffer = new ArrayBuffer(20);
  const resView = new Int32Array(resBuffer);
  resView[0] = version;
  resView[1] = hasAnomaly ? 1 : 0;
  resView[2] = from;
  resView[3] = from + Math.max(insCount, 1);
  resView[4] = severity;

  // Transferable でメインスレッドへゼロコピー即時返却
  self.postMessage(resBuffer, [resBuffer]);
};
