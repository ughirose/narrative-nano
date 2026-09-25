# Wasm SIMD 超高速推論コア仕様書

## 1. 概要
Narrative-Nano の推論エンジンは、C++ で記述され、WebAssembly 128-bit SIMD (`-msimd128`) 命令セットを活用してブラウザ内（Web Worker）でサブミリ秒推論を実行する。

## 2. ビルド & 最適化構成
- **コンパイラ**: Emscripten / Clang
- **フラグ構成**:
  - `-O3 -flto`: 最大限のインライン展開とリンク時最適化
  - `-msimd128`: Wasm SIMD 命令をフル活用した行列演算・内積計算
  - `-s STANDALONE_WASM=1`: ランタイム不要の独立した `.wasm` 生成
  - `-s INITIAL_MEMORY=16MB`: 静的バッファおよび推論ワークスペースの事前確保
- **エクスポートシンボル**:
  - `_init_model(const void* model_weights_ptr, size_t size)`
  - `_forward_step(const int32_t* token_seq, size_t len, float* output_logits)`
  - `_free_model()`
- **サイズ目標**:
  - バイナリサイズ: 262KB（目標上限 4MB を大幅に下回る軽量性を達成）
