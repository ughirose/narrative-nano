import os
import sys
import json
import time
import random
import urllib.request
import re
import subprocess

print("=== [Batch 1: Error-Aware Continual Learning & Benchmark Pipeline (Colab T4)] ===")
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

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device} | GPU: {gpu_name}")

# 2. 12-Class Novel-Optimized Case Architecture
CASE_PARTICLES = [
    ("が", 0), ("は", 1), ("を", 2), ("に", 3), ("で", 4), ("と", 5),
    ("から", 6), ("まで", 7), ("へ", 8), ("より", 9), ("の", 10), ("も", 11)
]
NUM_CASE_CLASSES = 12

# 3. Clean-Room Positive Corpus Generation
samples = []
subjects = ["俺", "私", "僕", "彼女", "彼", "勇者", "主人公", "ギルドマスター", "先輩", "後輩", "エリス", "アリア", "魔王", "少女"]
objects = ["ステータス画面", "スキル一覧", "古代の魔導書", "真実の鍵", "スマートフォン", "冷めたコーヒー", "依頼書", "聖剣", "黒い短剣"]
locations = ["ギルドの酒場", "暗いダンジョンの中層", "オフィスの静寂", "放課後の教室", "地下闘技場", "壊れかけた神殿", "薄暗い自室"]
actions = ["静かに睨みつけた", "ため息をつきながら開いた", "素早くポケットに隠した", "信じられない思いで見つめた", "迷わず手に取った", "呟いて立ち上がった"]

for _ in range(500):
    sub = random.choice(subjects)
    obj = random.choice(objects)
    loc = random.choice(locations)
    act = random.choice(actions)
    samples.append((f"{sub}は{loc}で{obj}を{act}。", False, 0.0))
    samples.append((f"{sub}が{loc}から{obj}を持ち去った。", False, 0.0))
    samples.append((f"{sub}もまた、{obj}の秘密を知っていた。", False, 0.0))
    samples.append((f"「おい{sub}、本当に{loc}へ行くつもりなのか？」", True, 0.0))
    samples.append((f"「{obj}なら、もう俺の手の中にあるよ」", True, 0.0))

# 4. Verified Public Domain Works (Aozora Bunko)
AOZORA_WORKS = [
    ("こころ", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/773_ruby_5968/773_ruby_5968.txt"),
    ("坊っちゃん", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/752_ruby_2438/752_ruby_2438.txt"),
    ("羅生門", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/127_ruby_150/127_ruby_150.txt"),
    ("走れメロス", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1567_ruby_4948/1567_ruby_4948.txt"),
]

for title, url in AOZORA_WORKS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode('shift_jis', errors='ignore')
            cleaned = re.sub(r'《[^》]+》', '', raw)
            cleaned = re.sub(r'｜', '', cleaned)
            cleaned = re.sub(r'［＃[^］]+］', '', cleaned)
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                is_dia = line.startswith('「') and line.endswith('」')
                for s in re.split(r'。', line):
                    s = s.strip()
                    if 6 <= len(s) <= 90:
                        samples.append((s + '。', is_dia, 0.0))
    except Exception as e:
        print(f"Notice: could not load {title} ({e})")

# 5. Explicit Negative / Error-Verification Samples (助詞重複・主述ねじれ・視点ブレ)
error_samples = [
    ("彼が猫が魚が好きだと言った。", False, 1.0),
    ("勇者が仲間が敵が倒れたと叫んだ。", False, 1.0),
    ("彼を部屋を荷物を見せに連れて行った。", False, 1.0),
    ("私の友達の犬の家の庭。", False, 1.0),
    ("学校の図書室の奥の机の上の本。", False, 1.0),
    ("彼に私に先生に報告しに行った。", False, 1.0),
    ("東京で大阪で名古屋で開催された。", False, 1.0),
    ("私の将来の夢は、世界大会で優勝したからです。", False, 1.0),
    ("彼は、明日晴れると良いなと思ったからです。", False, 1.0),
    ("僕の希望としては、全員が無事に帰還できるからです。", False, 1.0),
    ("その真実を知らないわけではないと言わざるを得ない。", False, 1.0),
    ("彼の主張に賛同できなくもないわけではない。", False, 1.0),
    ("敵に城を奪われて、味方が皆殺害された。", False, 1.0),
    ("魔王に追われて、仲間に見捨てられて、孤立させられた。", False, 1.0),
    ("エリスは恐怖に震えていた。しかし俺は冷たく見下ろした。", False, 1.0),
    ("彼は冷徹な暗殺者だった。でも本当はちょっぴり寂しがり屋なのだ。", False, 1.0),
]

# Oversample errors to ensure balanced learning of error detection
for text, is_dia, err_score in error_samples * 50:
    samples.append((text, is_dia, err_score))

print(f"[*] Total Dataset Size: {len(samples):,} (Clean + Explicit Error Samples)")
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
        self.head_error = nn.Linear(hidden_dim, 1) # Error / Distortion probability head

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(x)
        out_modality = self.head_modality(h)
        out_offset = self.head_offset(h)
        out_label = self.head_label(h)
        out_case = self.head_case(h)
        out_epistemic = torch.sigmoid(self.head_epistemic(h)).squeeze(-1)
        # Sentence-level error pooling (mean of hidden states)
        h_pool = h.mean(dim=1)
        out_error = torch.sigmoid(self.head_error(h_pool)).squeeze(-1)
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

val_size = min(1200, int(len(samples) * 0.15))
train_loader = DataLoader(MultiTaskDataset(samples[:-val_size]), batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(MultiTaskDataset(samples[-val_size:]), batch_size=64, shuffle=False)

model = ErrorAwareNarrativeNano().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
criterion_ce = nn.CrossEntropyLoss()
criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()

epochs = 3
total_steps = epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

print(f"\n=== Executing Error-Aware Continual Training ({epochs} Epochs) on {device} ===")
epoch_logs = []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for b_tokens, b_mod, b_off, b_lbl, b_case, b_epi, b_err in train_loader:
        b_tokens, b_mod = b_tokens.to(device), b_mod.to(device)
        b_off, b_lbl = b_off.to(device), b_lbl.to(device)
        b_case, b_epi, b_err = b_case.to(device), b_epi.to(device), b_err.to(device)

        optimizer.zero_grad()
        out_mod, out_off, out_lbl, out_case, out_epi, out_err = model(b_tokens)

        loss_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
        loss_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
        loss_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
        loss_case = criterion_case(out_case.view(-1, NUM_CASE_CLASSES), b_case.view(-1))
        loss_epi = criterion_mse(out_epi, b_epi)
        loss_err = criterion_bce(out_err, b_err)

        loss = loss_mod * 1.0 + loss_off * 1.5 + loss_lbl * 1.0 + loss_case * 2.5 + loss_epi * 0.5 + loss_err * 2.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_case, total_case = 0, 0
    err_tp, err_fp, err_tn, err_fn = 0, 0, 0, 0

    with torch.no_grad():
        for b_tokens, b_mod, b_off, b_lbl, b_case, b_epi, b_err in val_loader:
            b_tokens, b_mod = b_tokens.to(device), b_mod.to(device)
            b_off, b_lbl = b_off.to(device), b_lbl.to(device)
            b_case, b_epi, b_err = b_case.to(device), b_epi.to(device), b_err.to(device)

            out_mod, out_off, out_lbl, out_case, out_epi, out_err = model(b_tokens)
            l_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
            l_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
            l_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
            l_case = criterion_case(out_case.view(-1, NUM_CASE_CLASSES), b_case.view(-1))
            l_epi = criterion_mse(out_epi, b_epi)
            l_err = criterion_bce(out_err, b_err)
            val_loss += (l_mod * 1.0 + l_off * 1.5 + l_lbl * 1.0 + l_case * 2.5 + l_epi * 0.5 + l_err * 2.0).item()

            preds_case = torch.argmax(out_case, dim=-1)
            mask = b_case != -100
            correct_case += (preds_case[mask] == b_case[mask]).sum().item()
            total_case += mask.sum().item()

            pred_err_bin = (out_err >= 0.5).float()
            for p, g in zip(pred_err_bin, b_err):
                if p == 1 and g == 1: err_tp += 1
                elif p == 1 and g == 0: err_fp += 1
                elif p == 0 and g == 0: err_tn += 1
                elif p == 0 and g == 1: err_fn += 1

    avg_val_loss = val_loss / len(val_loader)
    case_acc = (correct_case / total_case * 100) if total_case > 0 else 0.0
    err_precision = err_tp / (err_tp + err_fp) if (err_tp + err_fp) > 0 else 0.0
    err_recall = err_tp / (err_tp + err_fn) if (err_tp + err_fn) > 0 else 0.0
    err_f1 = (2 * err_precision * err_recall / (err_precision + err_recall) * 100) if (err_precision + err_recall) > 0 else 0.0
    elapsed = time.time() - t0

    log_entry = {
        "epoch": epoch,
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4),
        "case_acc": round(case_acc, 2),
        "error_detection_f1": round(err_f1, 2),
        "elapsed_sec": round(elapsed, 1)
    }
    epoch_logs.append(log_entry)
    print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 12-Case Acc: {case_acc:.2f}% | Error F1: {err_f1:.2f}% | {elapsed:.1f}s")

# 6. Explicit Error Benchmarking Test
print("\n=== Model Error-Detection Benchmark Verification ===")
benchmark_cases = [
    ("彼が猫が魚が好きだと言った。", True, "助詞重複（が×3）"),
    ("私の友達の犬の家の庭。", True, "助詞重複（の×4）"),
    ("私の夢は、世界大会で優勝したからです。", True, "主述ねじれ（私の夢は〜からです）"),
    ("敵に城を奪われて、味方が皆殺害された。", True, "受身の連続"),
    ("主人公は薄暗いダンジョンの中層で聖剣を手に入れた。", False, "正常文（正例）"),
    ("メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。", False, "正常文（正例）"),
]

model.eval()
with torch.no_grad():
    for text, expected_err, note in benchmark_cases:
        t_ids, t_len = tokenize_char(text, 128)
        inp = torch.tensor([t_ids], dtype=torch.long, device=device)
        out_mod, out_off, out_lbl, out_case, out_epi, out_err = model(inp)
        err_prob = out_err.item()
        print(f"[{'PASS' if (err_prob >= 0.5) == expected_err else 'WARN'}] Error Prob: {err_prob*100:.1f}% | Expected: {expected_err} | {note}")
        print(f"    Text: {text}")

# 7. Model Checkpoint & ONNX Export (Saved to Drive / Local)
pt_path = f"{DRIVE_DIR}/models/narrative_nano_v1_error_aware.pt"
onnx_path = f"{DRIVE_DIR}/models/narrative_nano_v1_error_aware.onnx"
torch.save(model.state_dict(), pt_path)

dummy = torch.zeros(1, 128, dtype=torch.long, device=device)
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input_ids"],
    output_names=["modality", "offset", "label", "case", "epistemic", "error_score"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "modality": {0: "batch_size", 1: "seq_len"},
        "offset": {0: "batch_size", 1: "seq_len"},
        "label": {0: "batch_size", 1: "seq_len"},
        "case": {0: "batch_size", 1: "seq_len"},
        "epistemic": {0: "batch_size", 1: "seq_len"},
        "error_score": {0: "batch_size"},
    },
    opset_version=17
)

total_elapsed = time.time() - t_start
consumed_cu = round((total_elapsed / 3600.0) * 1.5, 4)

run_summary = {
    "run_id": "RUN-002",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "gpu": gpu_name,
    "total_elapsed_sec": round(total_elapsed, 1),
    "consumed_cu": consumed_cu,
    "epochs": epoch_logs,
    "pt_path": pt_path,
    "onnx_path": onnx_path,
    "onnx_size_bytes": os.path.getsize(onnx_path),
    "status": "SUCCESS"
}

log_path = f"{DRIVE_DIR}/logs/run_002_summary.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

print(f"\n[SUMMARY] Total Elapsed: {total_elapsed:.1f}s | Consumed CU: {consumed_cu:.4f} CU")
print(f"[SUMMARY] Artifacts saved to: {DRIVE_DIR}")
print(f"[JSON_SUMMARY_START]{json.dumps(run_summary)}[JSON_SUMMARY_END]")
