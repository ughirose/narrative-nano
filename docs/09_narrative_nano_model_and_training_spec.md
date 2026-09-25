# 09_narrative_nano_model_and_training_spec

## 【極小構文解析モデル & Wasm実行基盤】Narrative-Nano アーキテクチャ・学習・IPC詳細仕様書
- 文書ID: 09_narrative_nano_model_and_training_spec.md
- 対象プロダクト: Narrative-Nano Engine / WorldCraft Stage 1 Evaluator
- ステータス: 新規開発用確定仕様書
- 制定日: 2026-09-24


### 1. モデルアーキテクチャ概要
Narrative-Nanoは、Webブラウザ（Wasm SIMD / Web Worker）環境において、キーストロークに同期してサブミリ秒で動作する文芸特化型の極小マルチタスク自然言語処理モデルである。 形態素解析辞書やBPEサブワード分割器を一切排除し、文字レベル（Character-level）入力を直接受け取ることで、未定義語（<unk>）を原理的に発生させず、エディタ文字座標との1対1完全マッピングを実現する。
#### 1.1 基本ハイパーパラメータ
- パラメータ数: 約 5,834,059（約 5.8M）
- モデル構造: 6層 双方向 Transformer Encoder + マルチタスク射影ヘッド
- 入力語彙サイズ ($V$): 1,000 〜 2,048 文字（常用漢字、ひらがな、カタカナ、約物）
- 隠れ層次元 ($d_{\text{model}}$): 256
- アテンションヘッド数 ($H$): 4（$d_k = 64$）
- Feed-Forward中間拡大次元 ($d_{\text{ff}}$): 1024 (GELU活性化)
- 最大系列長 ($L_{\text{max}}$): 128 文字
- 正則化: Pre-LayerNorm, Dropout = 0.1
- 量子化ターゲット: INT8 対称量子化（ウェイトサイズ: 約 3.6MB 〜 4.2MB）


### 2. マルチタスク出力ヘッド数理モデル
バックボーン最終層の隠れ状態表現を $\mathbf{H} \in \mathbb{R}^{B \times L \times 256}$ とする。

- 様態分類ヘッド (ModalityHead): 各文字が発話（会話文）か地の文かを二値分類。 $$\hat{\mathbf{y}}i^{\text{mod}} = \mathrm{softmax}(\mathbf{W}{\text{mod}} h_i + b_{\text{mod}})$$
- 係り受け相対オフセットヘッド (OffsetHead): 各文字 $i$ から係り先主辞 $j$ への相対差分 $\Delta_i = j - i \in [-32, +32]$（計65クラス）を分類。$O(L)$ 系列ラベリング。 $$\hat{\mathbf{y}}i^{\text{offset}} = \mathrm{softmax}(\mathbf{W}{\text{offset}} h_i + b_{\text{offset}})$$
- 係り受け関係ラベルヘッド (LabelHead): 係り受け種別（8クラス: ROOT, D, APPOSITION, P, A, I, DEPENDENCY, pad）を分類。
- 動的バイアフィン述語項構造ヘッド (DynamicBiaffinePASHead): 主要10格（ガ、ガ２、ヲ、ニ、ト、デ、カラ、ヨリ、ヘ、マデ）について、述語文字 $i$ と項文字 $j$ の格関係成立確率を双線形内積によって算出。 $$\mathbf{S}_{c, i, j} = (h_i^{\text{pred}})^T \mathcal{U}_c , h_j^{\text{arg}}$$


### 3. Lock-free Ring Buffer IPC 仕様 (SharedArrayBuffer + Atomics)
#### 3.1 メモリレイアウト (16MB / 256 Pages)
- Page 0 (64 KB): 制御ブロック（64Bキャッシュライン分離アトミックポインタ）
- Page 1..64 (4 MB): Request Ring Buffer（メイン $\to$ Worker）
- Page 65..128 (4 MB): Response Ring Buffer（Worker $\to$ メイン）
- Page 129..256 (8 MB): Wasm CST Node Arena & Heap
#### 3.2 終端スキップパディング（Skip-to-Head Sentinel）
リングバッファ終端を跨ぐ分割コピーを排除するため、折り返し時に PACKET_SENTINEL_WRAP (0xFF) を書き込み、常に連続メモリ領域としてゼロコピー走査を保証する。
#### 3.3 自己修復ウォッチドッグ（Watchdog & Recovery）
- Worker推論時間が 50ms を超過した場合、ハングアップと判定して強制終了（terminate()）。
- アトミックポインタをリセットし、新規Workerを即時再起動してCold Resyncを実行。