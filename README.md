# Narrative-Nano — 超軽量クライアント側文芸推論モジュール

> **2026-11-23 現行ステータス**:
> - `main`（`752e6c9`, 2026-11-10）。`v18` リークフリーコーパス訓練モデル（INT8 ONNX 2.69 MB、C++/Wasm カーネル `dist/narrative_core.wasm` 262 KB、Wasm P50 推論 1.01 ms、139 tests PASS）を提供。
> - **限界と位置づけ**:
>   - **Plotailor / WorldCraft 向け**: `@worldcraft/narrative-nano` としてエディタ側の高速補助推論に引き続き有効。
>   - **USIR v5 / Nishiki 向け**: 無括弧会話の単独再現率が 26.67%（4/15）にとどまるため、精密なオフライン場面注釈（`sensor.v2`）はローカル `qwen2.5:14b`＋強モデルハイブリッド（`ughirose/usir` `scripts/sensor_runner/`）へ移行済みです。
> - ドキュメント索引は [docs/INDEX.md](docs/INDEX.md)、4プロジェクト統合正本は Google Drive `05_Nishiki_錦/00_INDEX_AND_SSOT/` を参照してください。

---

## 主要スクリプト・コマンド

```bash
pnpm run typecheck
pnpm run test
```
