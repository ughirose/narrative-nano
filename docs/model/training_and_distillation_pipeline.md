# Google Colab 知識蒸留 & 学習パイプライン

## 1. 概要
大規模言語モデル（Gemini 1.5 Pro / Claude 3.5 Sonnet）を教師モデルとして用い、文芸創作の時系列変化・格関係判定に特化した合成データセットを生成して極小モデルへ知識蒸留（Knowledge Distillation）を行うパイプライン。

## 2. パイプライン構成
1. **合成データセット生成 (`generate_temporal_dataset.py`)**:
   - 登場人物の属性変化（死亡、負傷、所属移動、アイテム所持状態の変遷）を含む文芸スニペットを生成。
   - 10 種の日本語格助詞（ガ, ヲ, ニ, デ, ト, ヨリ, カラ, マデ, ヘ, ノ）に対応するアノテーション付与。
2. **蒸留学習 (`distill_narrative_nano.py` / `model.py`)**:
   - 教師モデルのソフトラベル出力を Cross-Entropy + KL Divergence 損失で学習。
3. **ONNX / バイナリ変換 (`export_onnx.py`)**:
   - PyTorch -> ONNX へのエクスポートおよび重みバイナリの抽出。
   - Wasm 推論コアが直接メモリマップできるフラットバッファ形式へ変換。
