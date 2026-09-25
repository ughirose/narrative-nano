/**
 * WorldCraft Reference Asset: cm6-nano-worker-plugin.ts
 * 
 * CodeMirror 6 ViewPlugin & O(Delta) 差分トークン抽出ディスパッチャ
 * 
 * 参照構想文書: 20260921_【構想】CodeMirror6×WebWorkerフロントエンド実装設計
 */

import { ViewPlugin, ViewUpdate, EditorView } from '@codemirror/view';

export function simpleTokenize(text: string): number[] {
  const tokens: number[] = [];
  for (let i = 0; i < text.length; i++) {
    tokens.push(text.charCodeAt(i) % 16384);
  }
  return tokens;
}

export function createNanoWorkerPlugin(workerPath: string) {
  return ViewPlugin.fromClass(
    class {
      private worker: Worker;
      private docVersion = 0;

      constructor(private view: EditorView) {
        this.worker = new Worker(workerPath, { type: 'module' });

        this.worker.onmessage = (e: MessageEvent<ArrayBuffer>) => {
          const res = new Int32Array(e.data);
          const version = res[0];
          const hasAnomaly = res[1] === 1;
          const from = res[2];
          const to = res[3];
          const severity = res[4];

          if (version === this.docVersion && hasAnomaly) {
            console.log(`[WorldCraft CM6] 設定矛盾アラート検知: pos(${from}-${to}), severity: ${severity}`);
          }
        };
      }

      update(update: ViewUpdate) {
        if (!update.docChanged) return;

        this.docVersion++;
        const currentVersion = this.docVersion;

        let fromOffset = 0;
        let deletedText = '';
        let insertedText = '';

        update.changes.iterChanges((fromA, toA, fromB, toB, inserted) => {
          fromOffset = fromA;
          deletedText += update.startState.sliceDoc(fromA, toA);
          insertedText += inserted.toString();
        });

        const delTokens = simpleTokenize(deletedText);
        const insTokens = simpleTokenize(insertedText);

        const packetLength = 4 + delTokens.length + insTokens.length;
        const buffer = new ArrayBuffer(packetLength * 4);
        const view = new Int32Array(buffer);

        view[0] = currentVersion;
        view[1] = fromOffset;
        view[2] = delTokens.length;
        view[3] = insTokens.length;

        let offset = 4;
        for (let i = 0; i < delTokens.length; i++) view[offset++] = delTokens[i];
        for (let i = 0; i < insTokens.length; i++) view[offset++] = insTokens[i];

        this.worker.postMessage(buffer, [buffer]);
      }

      destroy() {
        this.worker.terminate();
      }
    }
  );
}
