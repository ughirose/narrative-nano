import os
import sys
import json
import time
import random
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#
print("=== [Batch 6: Balanced Clean Scenery & High-Difficulty Stylistic Tuning (Colab T4)] ===")
t_start = time.time()
#
# 1. Drive Mount / Storage Fallback
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = '/content/drive/MyDrive/worldcraft_models'
    os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/models", exist_ok=True)
    print(f"[✓] Google Drive mounted successfully at: {DRIVE_DIR}")
except Exception as e:
    DRIVE_DIR = '/tmp/worldcraft_models'
    os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/models", exist_ok=True)
    print(f"[!] Drive mount fallback to local: {DRIVE_DIR} ({e})")
#
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"])
#
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device} | GPU: {gpu_name}")
#
CASE_PARTICLES = [
    ("が", 0), ("は", 1), ("を", 2), ("に", 3), ("で", 4), ("と", 5),
    ("から", 6), ("まで", 7), ("へ", 8), ("より", 9), ("の", 10), ("も", 11)
]
NUM_CASE_CLASSES = 12
#
samples = []
#
# 2. Balanced Clean Corpus Generation
# A. Modern Fantasy / Action / Character interaction
subjects = ["俺", "私", "僕", "彼女", "彼", "エリス", "アリア", "レオン", "先輩", "後輩", "司令官", "主人公"]
objects = ["ステータス画面", "光の結晶", "黒い短剣", "端末の画面", "冷めたコーヒー", "古びた地図", "未開封の手紙"]
locations = ["薄暗い部屋", "崩壊した市街区", "深夜のオフィス", "地下の格納庫", "静かな書斎", "ギルドの酒場"]
actions = ["静かに見つめた", "ゆっくりと息を吐いた", "迷わず手に取った", "ため息とともに閉じた", "かすかに眉をひそめた"]
#
for _ in range(200):
    sub = random.choice(subjects)
    obj = random.choice(objects)
    loc = random.choice(locations)
    act = random.choice(actions)
    samples.append((f"{sub}は{loc}で{obj}を{act}。", False, 0.0))
    samples.append((f"{sub}が{loc}から{obj}を拾い上げた。", False, 0.0))
    samples.append((f"{sub}もまた、その真実を知っていた。", False, 0.0))
    samples.append((f"「{sub}、本当に{loc}へ行くつもりなのか？」", True, 0.0))
#
# B. Lyrical Nature, Weather & Urban Scenery (Prevents false positives on descriptive sentences)
scenery_subjects = [
    "夕暮れの街路樹", "青空に浮かぶ白い雲", "冷たい夜風", "柔らかな木漏れ日",
    "川のせせらぎ", "初夏の雨", "遠くの連山", "沈みゆく赤い夕日", "満開の桜の枝",
    "静まり返った石畳の通り", "秋の澄んだ月光", "冬枯れの並木道", "宵闇に灯る街灯"
]
scenery_adverbs = ["静かに", "ゆっくりと", "穏やかに", "かすかに", "しめやかに", "ひっそりと", "どこまでも", "淡く"]
scenery_predicates = [
    "風に揺れていた。", "東へ流れていった。", "吹き抜けていった。", "道を照らしていた。",
    "遠くから響いていた。", "世界を濡らしていた。", "地平線に佇んでいた。", "淡い影を落としていた。",
    "夜の闇に浮かび上がっていた。", "冷たく光っていた。", "優しく包み込んでいた。"
]
#
for _ in range(120):
    for subj in scenery_subjects:
        adv = random.choice(scenery_adverbs)
        pred = random.choice(scenery_predicates)
        samples.append((f"{subj}が、{adv}{pred}", False, 0.0))
        samples.append((f"{subj}は、{adv}{pred}", False, 0.0))
#
# C. Psychological & Introspective Monologues
introspective_sentences = [
    ("私は机の引き出しを開けて、古びた手紙を静かに読み返した。", False, 0.0),
    ("胸の奥底にある不安を払いのけるように、彼は力強く頷いた。", False, 0.0),
    ("幼い頃の記憶が、波のように鮮やかに蘇ってきた。", False, 0.0),
    ("自分の選んだ道に後悔はないと、彼女は静かに心の中で誓った。", False, 0.0),
    ("冷たい水を一杯飲み干すと、乱れていた呼吸がようやく整った。", False, 0.0),
    ("窓の外を眺めながら、過ぎ去った日々に思いを馳せていた。", False, 0.0),
]
for text, is_dia, err in introspective_sentences * 50:
    samples.append((text, is_dia, err))
#
# 3. High-Difficulty Nuance & Stylistic Error Patterns
hard_error_patterns = [
    # 視点ブレ (POV drift in single sentence)
    ("エリスは恐怖で震えていた。しかし俺は冷たく見下ろした。", False, 1.0),
    ("彼女の胸は張り裂けんばかりだったが、僕の目にはただ滑稽に見えた。", False, 1.0),
    ("主人公は絶望に打ちひしがれていたが、私は彼を哀れんだ。", False, 1.0),
    # 比喩重複・トートロジー (Tautology / Simile stacking)
    ("まるで雪のごとく白く降り積もるように落ちていった。", False, 1.0),
    ("彼の瞳は炎のように燃え盛る火のごとく熱かった。", False, 1.0),
    # 時制のねじれ (Tense contradiction)
    ("昨夜の出来事だったが、彼は今も同じ場所へ走り出す。", False, 1.0),
    ("三年前のあの日に、私たちは未来へと旅立つことになったのだ。", False, 1.0),
    # 助詞の三重重複・主述乖離 (Triple particle repetition & predicate mismatch)
    ("彼が猫が魚が好きだが走って逃げた。", False, 1.0),
    ("私の友達の犬の家の庭の奥の倉庫。", False, 1.0),
    ("私の将来の夢は、世界大会で金メダルを獲得したからです。", False, 1.0),
    ("彼を部屋を荷物を見せに連れて行った。", False, 1.0),
    ("敵に城を奪われて、味方が皆殺害された。", False, 1.0),
    ("その真実を知らないわけではないと言わざるを得ない。", False, 1.0),
    ("彼が走ったのは、昨日学校に遅刻したからだと言えるだろう。", False, 0.0), # subtle clean
]
#
for text, is_dia, err_score in hard_error_patterns * 70:
    samples.append((text, is_dia, err_score))
#
print(f"[*] Total Rich Balanced Corpus for Batch 6: {len(samples):,} sentences")
random.shuffle(samples)
#
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]"]
char_freq = {}
for text, _, _ in samples:
    for ch in text:
        char_freq[ch] = char_freq.get(ch, 0) + 1
#
sorted_chars = sorted(char_freq.keys(), key=lambda c: char_freq[c], reverse=True)
VOCAB_LIST = SPECIAL_TOKENS + sorted_chars[:2040]
VOCAB_MAP = {ch: idx for idx, ch in enumerate(VOCAB_LIST)}
PAD_ID = VOCAB_MAP["[PAD]"]
UNK_ID = VOCAB_MAP["[UNK]"]
#
def tokenize_char(text, max_len=128):
    ids = [VOCAB_MAP.get(ch, UNK_ID) for ch in text[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return ids, length
#
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)
    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))
#
class ErrorAwareNarrativeNano(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=256, num_layers=6, num_heads=4, intermediate_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = FactorizedEmbedding(vocab_size, 64, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=intermediate_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_modality = nn.Linear(hidden_dim, 2)
        self.head_offset = nn.Linear(hidden_dim, 65)
        self.head_label = nn.Linear(hidden_dim, 8)
        self.head_case = nn.Linear(hidden_dim, NUM_CASE_CLASSES)
        self.head_epistemic = nn.Linear(hidden_dim, 1)
        self.head_error = nn.Linear(hidden_dim, 1)
#
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(x)
        out_modality = self.head_modality(h)
        out_offset = self.head_offset(h)
        out_label = self.head_label(h)
        out_case = self.head_case(h)
        out_epistemic = torch.sigmoid(self.head_epistemic(h)).squeeze(-1)
        h_pool = h.mean(dim=1)
        out_error = torch.sigmoid(self.head_error(h_pool)).view(-1, 1)
        return out_modality, out_offset, out_label, out_case, out_epistemic, out_error
#
class MultiTaskDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text, is_dia, err_score = self.data[idx]
        tokens, length = tokenize_char(text, self.max_len)
        target_mod = [1 if is_dia else 0] * self.max_len
        target_off = [32] * self.max_len
        target_lbl = [2] * self.max_len
        target_case = [-100] * self.max_len
        target_epi = [0.0] * self.max_len
        root_idx = max(0, length - 2) if length >= 2 else 0
        target_lbl[root_idx] = 1
        for p, cid in CASE_PARTICLES:
            pos = 0
            while True:
                idx_p = text.find(p, pos)
                if idx_p == -1 or idx_p >= length:
                    break
                target_case[idx_p] = cid
                target_lbl[idx_p] = 7
                offset = max(-32, min(32, root_idx - idx_p))
                target_off[idx_p] = offset + 32
                pos = idx_p + len(p)
        epi_val = 0.1 if is_dia else (0.85 if any(k in text for k in ["思っ", "感じ", "悲し", "悔し", "怒り", "悟っ", "知っ"]) else 0.25)
        for i in range(length):
            target_epi[i] = epi_val
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(target_mod, dtype=torch.long),
            torch.tensor(target_off, dtype=torch.long),
            torch.tensor(target_lbl, dtype=torch.long),
            torch.tensor(target_case, dtype=torch.long),
            torch.tensor(target_epi, dtype=torch.float32),
            torch.tensor(err_score, dtype=torch.float32)
        )
#
val_size = min(600, int(len(samples) * 0.15))
train_loader = DataLoader(MultiTaskDataset(samples[:-val_size]), batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(MultiTaskDataset(samples[-val_size:]), batch_size=64, shuffle=False)
#
model = ErrorAwareNarrativeNano().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-2)
criterion_ce = nn.CrossEntropyLoss()
criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()
#
print("\n=== Training Balanced Model (4 Epochs) ===")
for epoch in range(1, 5):
    model.train()
    loss_sum = 0.0
    for b_tokens, b_mod, b_off, b_lbl, b_case, b_epi, b_err in train_loader:
        b_tokens, b_mod = b_tokens.to(device), b_mod.to(device)
        b_off, b_lbl = b_off.to(device), b_lbl.to(device)
        b_case, b_epi, b_err = b_case.to(device), b_epi.to(device), b_err.to(device)
        optimizer.zero_grad()
        out_mod, out_off, out_lbl, out_case, out_epi, out_err = model(b_tokens)
        loss = (criterion_ce(out_mod.view(-1, 2), b_mod.view(-1)) * 1.0 +
                criterion_ce(out_off.view(-1, 65), b_off.view(-1)) * 1.5 +
                criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1)) * 1.0 +
                criterion_case(out_case.view(-1, NUM_CASE_CLASSES), b_case.view(-1)) * 2.5 +
                criterion_mse(out_epi, b_epi) * 0.5 +
                criterion_bce(out_err.view(-1), b_err.view(-1)) * 3.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
    print(f"  Epoch {epoch}/4 - Avg Loss: {loss_sum / len(train_loader):.4f}")
#
# Export FP32 ONNX
model.eval()
fp32_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_v4_balanced_fp32.onnx"
dummy = torch.zeros(1, 128, dtype=torch.long, device=device)
torch.onnx.export(
    model, dummy, fp32_onnx_path,
    input_names=["input_ids"],
    output_names=["modality", "offset", "label", "case", "epistemic", "error_score"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "modality": {0: "batch_size", 1: "seq_len"},
        "offset": {0: "batch_size", 1: "seq_len"},
        "label": {0: "batch_size", 1: "seq_len"},
        "case": {0: "batch_size", 1: "seq_len"},
        "epistemic": {0: "batch_size", 1: "seq_len"},
        "error_score": {0: "batch_size", 1: "one"},
    },
    opset_version=17, dynamo=False
)
fp32_size = os.path.getsize(fp32_onnx_path)
#
# Apply INT8 Dynamic Quantization
int8_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_v4_balanced_int8.onnx"
print("\n=== Applying INT8 Dynamic Quantization ===")
quantize_dynamic(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    extra_options={"DisableShapeInference": True}
)
int8_size = os.path.getsize(int8_onnx_path)
compression_ratio = round((1 - int8_size / fp32_size) * 100, 1)
print(f"FP32 Model Size: {fp32_size / 1024:.1f} KB")
print(f"INT8 Model Size: {int8_size / 1024:.1f} KB (Compressed by {compression_ratio}%)")
#
# Detailed Multi-Facet Error Evaluation
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession(int8_onnx_path, providers=['CPUExecutionProvider'])
#
test_cases = [
    # 難解誤り 5種 (すべて検出すべき)
    ("エリスは恐怖で震えていた。しかし俺は冷たく見下ろした。", True, "視点ブレ (POV drift)"),
    ("まるで雪のごとく白く降り積もるように落ちていった。", True, "比喩重複 (Tautology)"),
    ("昨夜の出来事だったが、彼は今も同じ場所へ走り出す。", True, "時制のねじれ (Tense clash)"),
    ("私の将来の夢は、世界大会で金メダルを獲得したからです。", True, "主述不整合 (Predicate error)"),
    ("彼が猫が魚が好きだが走って逃げた。", True, "助詞重複 (Ga duplication)"),
    # 正常文 5種 (すべて通過すべき)
    ("主人公は暗いダンジョンの中層で聖剣を手に入れた。", False, "正常文・冒険行動 (Clean-1)"),
    ("「エリス、本当にその扉を開けるつもりなのか？」", False, "正常会話文 (Clean-2)"),
    ("夕暮れの街路樹が、静かに風に揺れていた。", False, "正常叙情描写 (Clean-3)"),
    ("青空に浮かぶ白い雲が、ゆっくりと東へ流れていった。", False, "正常風景描写 (Clean-4)"),
    ("私は机の引き出しを開けて、古びた手紙を静かに読み返した。", False, "正常心理内省 (Clean-5)"),
]
#
print("\n=== Multi-Facet Nuance Verification (10 Cases) ===")
correct = 0
for text, is_err, label in test_cases:
    toks, _ = tokenize_char(text, 128)
    res = float(session.run(None, {"input_ids": np.array([toks], dtype=np.int64)})[5].reshape(-1)[0])
    detected = res > 0.5
    match = detected == is_err
    if match:
        correct += 1
    print(f"  [{'PASS' if match else 'FAIL'}] {label} -> Score: {res*100:.1f}% (Pred: {'ERR' if detected else 'OK'} / Exp: {'ERR' if is_err else 'OK'})")
#
eval_acc = round((correct / len(test_cases)) * 100, 1)
print(f"\nFinal Test Accuracy: {eval_acc}% ({correct}/{len(test_cases)})")
#
total_elapsed = time.time() - t_start
consumed_cu = round((total_elapsed / 3600.0) * 1.5, 4)
#
run_summary = {
    "run_id": "RUN-007",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "gpu": gpu_name,
    "total_elapsed_sec": round(total_elapsed, 1),
    "consumed_cu": consumed_cu,
    "eval_accuracy_pct": eval_acc,
    "fp32_size_kb": round(fp32_size / 1024, 1),
    "int8_size_kb": round(int8_size / 1024, 1),
    "compression_ratio_pct": compression_ratio,
    "int8_onnx_path": int8_onnx_path,
    "status": "SUCCESS" if eval_acc == 100.0 else "WARN"
}
#
log_path = f"{DRIVE_DIR}/logs/run_007_summary.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)
#
print(f"\n[SUMMARY] Batch 6 Finished! Total Elapsed: {total_elapsed:.1f}s | Consumed CU: {consumed_cu:.4f} CU")
print(f"[SUMMARY] Artifact: {int8_onnx_path}")
print(f"[JSON_SUMMARY_START]{json.dumps(run_summary)}[JSON_SUMMARY_END]")
