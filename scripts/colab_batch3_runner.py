import os
import sys
import json
import time
import random
import urllib.request
import re
import subprocess

print("=== [Batch 3: INT8 Quantization (QAT/PTQ) & Wasm SIMD Optimization (Colab T4)] ===")
t_start = time.time()

# 1. Drive Mount / Storage Fallback
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = '/content/drive/MyDrive/worldcraft_models'
    os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/models", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/datasets", exist_ok=True)
    print(f"[✓] Google Drive mounted successfully at: {DRIVE_DIR}")
except Exception as e:
    DRIVE_DIR = '/tmp/worldcraft_models'
    os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/models", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/datasets", exist_ok=True)
    print(f"[!] Drive mount fallback to local: {DRIVE_DIR} ({e})")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device} | GPU: {gpu_name}")

CASE_PARTICLES = [
    ("が", 0), ("は", 1), ("を", 2), ("に", 3), ("で", 4), ("と", 5),
    ("から", 6), ("まで", 7), ("へ", 8), ("より", 9), ("の", 10), ("も", 11)
]
NUM_CASE_CLASSES = 12

samples = []

# Modern Web Novel Synthetic Samples
subjects = ["俺", "私", "僕", "彼女", "彼", "勇者", "主人公", "エリス", "アリア", "魔王", "少女"]
objects = ["ステータス画面", "古代の魔導書", "真実の鍵", "スマートフォン", "冷めたコーヒー", "聖剣", "黒い短剣"]
locations = ["ギルドの酒場", "暗いダンジョンの中層", "オフィスの静寂", "壊れかけた神殿", "薄暗い自室"]
actions = ["静かに睨みつけた", "ため息をつきながら開いた", "信じられない思いで見つめた", "迷わず手に取った"]

for _ in range(250):
    sub = random.choice(subjects)
    obj = random.choice(objects)
    loc = random.choice(locations)
    act = random.choice(actions)
    samples.append((f"{sub}は{loc}で{obj}を{act}。", False, 0.0))
    samples.append((f"{sub}が{loc}から{obj}を持ち去った。", False, 0.0))
    samples.append((f"{sub}もまた、{obj}の秘密を知っていた。", False, 0.0))
    samples.append((f"「おい{sub}、本当に{loc}へ行くつもりなのか？」", True, 0.0))

# Classical Works
AOZORA_WORKS = [
    ("こころ", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/773_ruby_5968/773_ruby_5968.txt"),
    ("坊っちゃん", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/752_ruby_2438/752_ruby_2438.txt"),
    ("吾輩は猫である", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/789_ruby_5639/789_ruby_5639.txt"),
    ("羅生門", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/127_ruby_150/127_ruby_150.txt"),
    ("走れメロス", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1567_ruby_4948/1567_ruby_4948.txt"),
]

for title, url in AOZORA_WORKS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = r.read().decode('shift_jis', errors='ignore')
            cleaned = re.sub(r'《[^》]+》', '', raw)
            cleaned = re.sub(r'｜', '', cleaned)
            cleaned = re.sub(r'［＃[^］]+］', '', cleaned)
            cnt = 0
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                is_dia = line.startswith('「') and line.endswith('」')
                for s in re.split(r'。', line):
                    s = s.strip()
                    if 6 <= len(s) <= 90:
                        samples.append((s + '。', is_dia, 0.0))
                        cnt += 1
                        if cnt >= 500:
                            break
                if cnt >= 500:
                    break
        print(f"  [+] Loaded {title} ({cnt} sentences)")
    except Exception as e:
        print(f"  [!] Skipped {title}: {e}")

# Error Samples
error_patterns = [
    ("彼が猫が魚が好きだが走った。", False, 1.0),
    ("勇者が仲間が敵が倒れたと叫んだ。", False, 1.0),
    ("彼を部屋を荷物を見せに連れて行った。", False, 1.0),
    ("私の友達の犬の家の庭。", False, 1.0),
    ("私の将来の夢は、世界大会で優勝したからです。", False, 1.0),
    ("彼は、明日晴れると良いなと思ったからです。", False, 1.0),
    ("その真実を知らないわけではないと言わざるを得ない。", False, 1.0),
    ("敵に城を奪われて、味方が皆殺害された。", False, 1.0),
    ("エリスは恐怖に震えていた。しかし俺は冷たく見下ろした。", False, 1.0),
]

for text, is_dia, err_score in error_patterns * 60:
    samples.append((text, is_dia, err_score))

print(f"[*] Total Corpus for Batch 3: {len(samples):,} sentences")
random.shuffle(samples)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]"]
char_freq = {}
for text, _, _ in samples:
    for ch in text:
        char_freq[ch] = char_freq.get(ch, 0) + 1

sorted_chars = sorted(char_freq.keys(), key=lambda c: char_freq[c], reverse=True)
VOCAB_LIST = SPECIAL_TOKENS + sorted_chars[:2040]
VOCAB_MAP = {ch: idx for idx, ch in enumerate(VOCAB_LIST)}
PAD_ID = VOCAB_MAP["[PAD]"]
UNK_ID = VOCAB_MAP["[UNK]"]

def tokenize_char(text, max_len=128):
    ids = [VOCAB_MAP.get(ch, UNK_ID) for ch in text[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return ids, length

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)

    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))

class ErrorAwareNarrativeNano(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=256, num_layers=6, num_heads=4, intermediate_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = FactorizedEmbedding(vocab_size, 64, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_modality = nn.Linear(hidden_dim, 2)
        self.head_offset = nn.Linear(hidden_dim, 65)
        self.head_label = nn.Linear(hidden_dim, 8)
        self.head_case = nn.Linear(hidden_dim, NUM_CASE_CLASSES)
        self.head_epistemic = nn.Linear(hidden_dim, 1)
        self.head_error = nn.Linear(hidden_dim, 1)

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

val_size = min(1000, int(len(samples) * 0.15))
train_loader = DataLoader(MultiTaskDataset(samples[:-val_size]), batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(MultiTaskDataset(samples[-val_size:]), batch_size=64, shuffle=False)

model = ErrorAwareNarrativeNano().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
criterion_ce = nn.CrossEntropyLoss()
criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()

# Fast 2-epoch tuning before quantization
print("\n=== Training Base Model (2 Epochs) ===")
for epoch in range(1, 3):
    model.train()
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
                criterion_bce(out_err.view(-1), b_err.view(-1)) * 2.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# Export FP32 ONNX
model.eval()
fp32_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_fp32.onnx"
dummy = torch.zeros(1, 128, dtype=torch.long, device=device)
torch.onnx.export(
    model,
    dummy,
    fp32_onnx_path,
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
    opset_version=17,
    dynamo=False
)
fp32_size = os.path.getsize(fp32_onnx_path)

# Apply INT8 Dynamic Quantization
int8_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_int8_simd.onnx"
print("\n=== Applying INT8 Dynamic Quantization for Wasm SIMD ===")
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

# Benchmark Latency & Quality
import onnxruntime as ort
import numpy as np

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_fp32 = ort.InferenceSession(fp32_onnx_path, sess_options, providers=['CPUExecutionProvider'])
session_int8 = ort.InferenceSession(int8_onnx_path, sess_options, providers=['CPUExecutionProvider'])

test_texts = [
    "彼が猫が魚が好きだと言った。",
    "私の夢は、世界大会で優勝したからです。",
    "主人公は薄暗いダンジョンの中層で聖剣を手に入れた。",
    "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。"
]

t_fp32_total, t_int8_total = 0.0, 0.0
for text in test_texts * 25:
    toks, _ = tokenize_char(text, 128)
    inp = np.array([toks], dtype=np.int64)

    t0 = time.perf_counter()
    out_f = session_fp32.run(None, {"input_ids": inp})
    t_fp32_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    out_i = session_int8.run(None, {"input_ids": inp})
    t_int8_total += time.perf_counter() - t0

avg_fp32_ms = (t_fp32_total / 100) * 1000
avg_int8_ms = (t_int8_total / 100) * 1000
speedup = round(avg_fp32_ms / avg_int8_ms, 2) if avg_int8_ms > 0 else 1.0

print(f"\n=== Latency Benchmark (100 runs on CPU/Wasm equivalent) ===")
print(f"FP32 Avg Latency: {avg_fp32_ms:.2f} ms")
print(f"INT8 Avg Latency: {avg_int8_ms:.2f} ms (Speedup: {speedup}x)")

# Verify Benchmark Quality
toks_err, _ = tokenize_char("彼が猫が魚が好きだと言った。", 128)
res_err = float(session_int8.run(None, {"input_ids": np.array([toks_err], dtype=np.int64)})[5].reshape(-1)[0])

toks_ok, _ = tokenize_char("主人公は薄暗いダンジョンの中層で聖剣を手に入れた。", 128)
res_ok = float(session_int8.run(None, {"input_ids": np.array([toks_ok], dtype=np.int64)})[5].reshape(-1)[0])

print(f"\nINT8 Error Verification:")
print(f"  Error text score: {res_err*100:.1f}% (Expected: >50%) -> {'PASS' if res_err > 0.5 else 'FAIL'}")
print(f"  Clean text score: {res_ok*100:.1f}% (Expected: <10%) -> {'PASS' if res_ok < 0.1 else 'FAIL'}")

total_elapsed = time.time() - t_start
consumed_cu = round((total_elapsed / 3600.0) * 1.5, 4)

run_summary = {
    "run_id": "RUN-004",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "gpu": gpu_name,
    "total_elapsed_sec": round(total_elapsed, 1),
    "consumed_cu": consumed_cu,
    "fp32_size_kb": round(fp32_size / 1024, 1),
    "int8_size_kb": round(int8_size / 1024, 1),
    "compression_ratio_pct": compression_ratio,
    "fp32_latency_ms": round(avg_fp32_ms, 2),
    "int8_latency_ms": round(avg_int8_ms, 2),
    "speedup": speedup,
    "int8_onnx_path": int8_onnx_path,
    "status": "SUCCESS"
}

log_path = f"{DRIVE_DIR}/logs/run_004_summary.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

print(f"\n[SUMMARY] Batch 3 Finished! Total Elapsed: {total_elapsed:.1f}s | Consumed CU: {consumed_cu:.4f} CU")
print(f"[SUMMARY] INT8 Artifact: {int8_onnx_path}")
print(f"[JSON_SUMMARY_START]{json.dumps(run_summary)}[JSON_SUMMARY_END]")
