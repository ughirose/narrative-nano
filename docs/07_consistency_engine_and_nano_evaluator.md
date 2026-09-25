# 07_consistency_engine_and_nano_evaluator

## 【世界観矛盾検知 & 極小判定モデル】WorldCraft 整合性検証仕様
### 1. 概要と基本思想
創作者の自由な発想を妨げない「スキーマの柔軟性」と、論理的破綻を自動検知するための「計算可能な厳密性」を両立させるため、コンパイラのようなブロッキング保存ではなく、コードレビューのLinter（静的解析）に近い思想で設計されています。作家が意図した例外（叙述トリック、時間遡行、不死者）にはルール抑制アノテーション（suppress_rules: ["temporal_overlap"]）を付与できる設計です。
### 2. 検知対象となる論理矛盾の5大分類
- 時系列矛盾（Temporal）:
  - 生没年と行動の逆転：イベント参加時期が人物の生誕（start）前、または死亡（end）後にある。
  - 年代逆転：親子の年齢逆転、前世代の出来事との矛盾。
- 因果・世代矛盾（Causal）:
  - 因果ループ：AがBの原因であり、BがAの原因である循環参照。
  - 伏線デッドロック：謎Aの解明が謎Bの前提であり、謎Bの解明が謎Aの前提である循環。
- 空間・移動矛盾（Spatial）:
  - 同一時間帯の多重存在：重複する日時に物理的に離れた複数イベントに参加している。
  - 移動速度の物理的限界突破：2地点間の移動日数が地形・交通手段の限界速度を超過。
- トポロジー・包含矛盾（Topological）:
  - 地理・組織の循環包含：地域Aが地域Bを含み、地域Bが地域Cを含み、地域Cが地域Aを含んでいる矛盾。
- 参照不整合（Referential / POV）:
  - 孤立・未定義スタブの放置：本文内で [[未作成エンティティ]] のまま長期放置。
  - 認知の漏洩（Epistemic Leak）: 視点人物（POV）が作中年代で知り得ない情報を語っている。
### 3. クライアント側 Property Graph データ構造
{
  "graph_version": "1.0.0",
  "project_id": "proj_98234",
  "nodes": [
    {
      "id": "char_alicia",
      "type": "character",
      "name": "アリシア",
      "temporal": { "birth": 1020, "death": 1085 },
      "tags": ["royal", "mage"],
      "meta": { "location_ref": "loc_capital" }
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "char_alicia",
      "target": "event_great_war",
      "relation": "participated_in",
      "temporal": { "start": 1011, "end": 1012 },
      "properties": { "role": "commander", "certainty": "established" }
    }
  ]
}
### 4. 大規模LLMからの知識蒸留（Knowledge Distillation）パイプライン
フロンティア級LLM（Claude 3.5 Sonnet / Qwen 2.5 72B）を教師モデルとし、極小モデル（Narrative-Nano）へ蒸留します。
- ステップ 1: コーパス疑似ラベリング: Web小説・パブリックドメイン原稿（約50万エピソード）からBIO境界・格関係・発話者・秘匿フラグを自動生成。
- ステップ 2: ロジット抽出 (Soft Target生成): 教師モデルの中間チェックポイントからSoft Probabilitiesを事前計算。
- ステップ 3: 複合蒸留学習 (PyTorch): Loss = α * L_task(ハードラベル) + β * L_KD(KL情報量) + γ * L_rep(隠れ層コサイン類似度)
- ステップ 4: 量子化・ONNX変換: QAT（Quantization Aware Training）を経て INT8 量子化し、約3.8MBのONNXバイナリへ出力。
### 5. Wasm SIMD vs WebGPU のベンチマークと選定基準
- 短文タイピング判定（数十〜100文字）では、GPUカーネルディスパッチのオーバーヘッド（0.2〜0.5ms）やデータ転送コストが推論時間を上回る。
- 結論: 単文タイピングのリアルタイム判定には、呼び出しオーバーヘッドが数マイクロ秒で完結する Wasm SIMD (CPU直接実行) を採用（推論時間 <0.05ms、全体レイテンシ 0.3〜0.8ms目標）。
- NarrativeNano 5.8M極小文字レベルモデルのアーキテクチャ（Transformer Encoder ＋ DynamicBiaffinePASHeadによる述語項構造・係り受け抽出）
- ミニマルペア合成データ境界例学習と自由間接話法（FID）の主観性スコア判定
- ゼロデータ保持（Zero-Data Retention）によるユーザー原稿保護と逆淘汰防止
- SharedArrayBufferとAtomicsによるSPSC Lock-free Ring Buffer IPC仕様（64Bキャッシュライン分離、終端スキップSentinel）
- WatchdogとWorker自動復旧（自己修復回路）アーキテクチャ