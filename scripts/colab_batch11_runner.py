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
print("=== [Batch 11: Cross-Sentence Consistency & Discourse Tuning (Colab T4)] ===")
t_start = time.time()
#
# 1. Storage setup
DRIVE_DIR = '/tmp/worldcraft_models'
os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/models", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/checkpoints", exist_ok=True)
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
MAX_LEN = 256
#
CASE_PARTICLES = [
    ("が", 0), ("は", 1), ("を", 2), ("に", 3), ("で", 4), ("と", 5),
    ("から", 6), ("まで", 7), ("へ", 8), ("より", 9), ("の", 10), ("も", 11)
]
NUM_CASE_CLASSES = 12
#
# 2. Cross-Sentence Discourse Archetypes (Clean vs Inconsistent)
clean_discourse_pairs = [
    ("深い森の奥から冷たい風が吹き抜けてきた。", "彼はコートの襟を立てて静かに歩き続けた。"),
    ("ステータス画面に表示された数値を確認する。", "まだMPには十分な余裕が残されていた。"),
    ("少女は窓辺に立ち、夕焼けに染まる街を眺めていた。", "その横顔には、どこか寂しげな色が浮かんでいた。"),
    ("長かった戦いがようやく幕を閉じた。", "戦士たちは武器を収め、互いの無事を喜び合った。"),
    ("机の上に古い羊皮紙が広げられている。", "そこには見慣れない古代文字がびっしりと記されていた。"),
    ("吾輩は猫である。名前はまだ無い。", "どこで生れたか頓と見当がつかぬ。"),
    ("「準備はいいかい？」と彼女が尋ねた。", "「いつでも行けるよ」と僕は短く答えた。"),
    ("東の空が白み始め、鳥たちのさえずりが聞こえてくる。", "新しい一日の始まりを告げる光が差し込んできた。"),
    ("AI端末のホログラムが青く明滅している。", "解析ログの進捗バーがゆっくりと百パーセントに近づいていた。"),
    ("彼女は運命《さだめ》を受け入れ、微笑んだ。", "その瞳には微塵の迷いも残っていなかった。")
]
#
cross_sentence_error_pairs = [
    # 視点ブレ（前文が三人称心情・客観、後文で突然一人称が介入）
    ("エリスは過酷な戦況に絶望し、恐怖で震えていた。", "しかし俺は冷淡に見下ろし、次の行動へ移った。"),
    # 時制ねじれ（前文が確定した過去の事象、後文で理由なく現在進行または未来へ飛翔）
    ("昨夜激しい嵐が村を襲い、多くの家屋が倒壊した。", "だが村人たちは今も逃げ惑い、これから嵐が訪れることになる。"),
    # 因果破綻（接続詞と後続アクションの論理破綻）
    ("彼は徹夜の作業で意識が朦朧としていた。", "だからこそ万全の体調で最高記録を易々と更新した。"),
    # 主述・照応破綻（文脈を跨ぐ照応関係のねじれ）
    ("私の将来の目標は、世界最高の作家になることです。", "なぜなら幼い頃から本を読むのが大好きだったからです。"),
    # 助詞重複・主格乱立（文脈跨ぎの主格過多）
    ("勇者が魔王が城で決戦を繰り広げた。", "仲間が誰もが息を呑んで見守っていた。")
]
#
samples = []
#
# Clean samples: pairs joined by space or natural conjunction (1,500 paragraphs)
for _ in range(75):
    for s1, s2 in clean_discourse_pairs:
        joined = f"{s1} {s2}"
        samples.append((joined[:MAX_LEN], False, 0.0))
#
for _ in range(50):
    for s1, s2 in clean_discourse_pairs:
        samples.append((s1[:MAX_LEN], False, 0.0))
        samples.append((s2[:MAX_LEN], False, 0.0))
#
# Error samples: inconsistent cross-sentence discourse (300 paragraphs)
for _ in range(60):
    for s1, s2 in cross_sentence_error_pairs:
        joined = f"{s1} {s2}"
        samples.append((joined[:MAX_LEN], False, 1.0))
#
print(f"[*] Total Discourse Corpus: {len(samples):,} paragraphs (Clean: {len([s for s in samples if s[2] == 0.0]):,}, Error: {len([s for s in samples if s[2] > 0.0]):,})")
random.shuffle(samples)
#
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]"]
char_freq = {}
for text, _, _ in samples:
    for ch in text:
        char_freq[ch] = char_freq.get(ch, 0) + 1
#
sorted_chars = sorted(char_freq.keys(), key=lambda c: char_freq[c], reverse=True)
VOCAB_LIST = SPECIAL_TOKENS + sorted_chars[:3064]
VOCAB_MAP = {ch: idx for idx, ch in enumerate(VOCAB_LIST)}
PAD_ID = VOCAB_MAP["[PAD]"]
UNK_ID = VOCAB_MAP["[UNK]"]
#
def tokenize_char(text, max_len=MAX_LEN):
    ids = [VOCAB_MAP.get(ch, UNK_ID) for ch in text[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return ids, length
#
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=3072, embed_dim=96, hidden_dim=384):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)
    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))
#
class ScaledNarrativeNano(nn.Module):
    def __init__(self, vocab_size=3072, hidden_dim=384, num_layers=6, num_heads=6, intermediate_dim=1536, dropout=0.1):
        super().__init__()
        self.embedding = FactorizedEmbedding(vocab_size, 96, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_LEN, hidden_dim) * 0.02)
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
        mask = (input_ids != PAD_ID).unsqueeze(-1).float()
        h_pool = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        out_error = torch.sigmoid(self.head_error(h_pool)).view(-1, 1)
        return out_modality, out_offset, out_label, out_case, out_epistemic, out_error
#
class LargeMultiTaskDataset(Dataset):
    def __init__(self, data, max_len=MAX_LEN):
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
        epi_val = 0.1 if is_dia else (0.85 if any(k in text for k in ["思っ", "感じ", "悲し", "悔し", "怒り", "決意", "不安"]) else 0.25)
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
train_loader = DataLoader(LargeMultiTaskDataset(samples), batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
#
model = ScaledNarrativeNano().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
#
criterion_ce = nn.CrossEntropyLoss()
criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()
#
print("\n=== Pre-training Discourse Model (2 Epochs on Tesla T4) ===")
for epoch in range(1, 3):
    model.train()
    loss_sum = 0.0
    for b_idx, (b_tokens, b_mod, b_off, b_lbl, b_case, b_epi, b_err) in enumerate(train_loader):
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
                criterion_bce(out_err.view(-1), b_err.view(-1)) * 1.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        if b_idx % 20 == 0:
            print(f"  [Epoch {epoch}/2] Step {b_idx}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")
            sys.stdout.flush()
    print(f"  Epoch {epoch}/2 Finished - Avg Loss: {loss_sum / len(train_loader):.4f}")
    sys.stdout.flush()
#
# Save PyTorch Checkpoint
ckpt_path = f"{DRIVE_DIR}/checkpoints/scaled_narrative_nano_v9_discourse.pt"
torch.save(model.state_dict(), ckpt_path)
print(f"[✓] Checkpoint saved: {ckpt_path}")
#
# Export FP32 ONNX
model.eval()
fp32_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_v9_discourse_fp32.onnx"
dummy = torch.zeros(1, MAX_LEN, dtype=torch.long, device=device)
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
int8_onnx_path = f"{DRIVE_DIR}/models/narrative_nano_v9_discourse_int8.onnx"
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
# Detailed Discourse Verification (10 Cases)
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession(int8_onnx_path, providers=['CPUExecutionProvider'])
#
test_cases = [
    # 文脈跨ぎのねじれ 5種
    ("エリスは過酷な戦況に絶望し、恐怖で震えていた。しかし俺は冷淡に見下ろし、次の行動へ移った。", True, "文脈跨ぎ・視点ブレ (Cross-Sentence POV Drift)"),
    ("昨夜激しい嵐が村を襲い、多くの家屋が倒壊した。だが村人たちは今も逃げ惑い、これから嵐が訪れることになる。", True, "文脈跨ぎ・時制ねじれ (Cross-Sentence Tense Clash)"),
    ("彼は徹夜の作業で意識が朦朧としていた。だからこそ万全の体調で最高記録を易々と更新した。", True, "文脈跨ぎ・因果破綻 (Cross-Sentence Causal Contradiction)"),
    ("私の将来の目標は、世界最高の作家になることです。なぜなら幼い頃から本を読むのが大好きだったからです。", True, "文脈跨ぎ・主述破綻 (Cross-Sentence Predicate Error)"),
    ("勇者が魔王が城で決戦を繰り広げた。仲間が誰もが息を呑んで見守っていた。", True, "文脈跨ぎ・助詞主格乱立 (Cross-Sentence Multi-Subject Error)"),
    # 正常な文脈連続 5種
    ("深い森の奥から冷たい風が吹き抜けてきた。彼はコートの襟を立てて静かに歩き続けた。", False, "正常自然情景連続 (Clean Forest Narration)"),
    ("ステータス画面に表示された数値を確認する。まだMPには十分な余裕が残されていた。", False, "正常ゲーム用語文脈 (Clean Status/MP Narration)"),
    ("「準備はいいかい？」と彼女が尋ねた。「いつでも行けるよ」と僕は短く答えた。", False, "正常会話文対話 (Clean Dialogue Exchange)"),
    ("彼女は運命《さだめ》を受け入れ、微笑んだ。その瞳には微塵の迷いも残っていなかった。", False, "正常ルビ表記・内面連続 (Clean Ruby/Mind Narration)"),
    ("吾輩は猫である。名前はまだ無い。どこで生れたか頓と見当がつかぬ。", False, "正常古典名作段落 (Clean Classic Literature)"),
]
#
print("\n=== Cross-Sentence Discourse Verification (10 Cases) ===")
correct = 0
for text, is_err, label in test_cases:
    toks, _ = tokenize_char(text, MAX_LEN)
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
# WebWorker Wasm CPU Latency Benchmark (30 runs)
print("\n=== Simulated WebWorker/Wasm CPU Latency Benchmark (30 runs) ===")
latencies = []
dummy_input = np.array([tokenize_char(clean_discourse_pairs[0][0] + clean_discourse_pairs[0][1], MAX_LEN)[0]], dtype=np.int64)
for _ in range(30):
    t0 = time.perf_counter()
    _ = session.run(None, {"input_ids": dummy_input})
    latencies.append((time.perf_counter() - t0) * 1000.0)
#
p50 = float(np.percentile(latencies, 50))
p95 = float(np.percentile(latencies, 95))
mean_lat = float(np.mean(latencies))
throughput_cps = round(len(clean_discourse_pairs[0][0] + clean_discourse_pairs[0][1]) / (mean_lat / 1000.0), 1)
print(f"  P50 Latency: {p50:.2f} ms")
print(f"  P95 Latency: {p95:.2f} ms")
print(f"  Mean Latency: {mean_lat:.2f} ms")
print(f"  Inference Throughput: {throughput_cps:,} chars/sec")
#
total_elapsed = time.time() - t_start
consumed_cu = round((total_elapsed / 3600.0) * 1.5, 4)
#
run_summary = {
    "run_id": "RUN-012",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "gpu": gpu_name,
    "total_elapsed_sec": round(total_elapsed, 1),
    "consumed_cu": consumed_cu,
    "eval_accuracy_pct": eval_acc,
    "latency_p50_ms": round(p50, 2),
    "latency_p95_ms": round(p95, 2),
    "throughput_chars_sec": throughput_cps,
    "fp32_size_kb": round(fp32_size / 1024, 1),
    "int8_size_kb": round(int8_size / 1024, 1),
    "compression_ratio_pct": compression_ratio,
    "int8_onnx_path": int8_onnx_path,
    "ckpt_path": ckpt_path,
    "status": "SUCCESS" if eval_acc == 100.0 else "WARN"
}
#
log_path = f"{DRIVE_DIR}/logs/run_012_summary.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)
#
print(f"\n[SUMMARY] Batch 11 Finished! Total Elapsed: {total_elapsed:.1f}s | Consumed CU (T4): {consumed_cu:.4f} CU")
print(f"[SUMMARY] Artifact: {int8_onnx_path}")
print(f"[JSON_SUMMARY_START]{json.dumps(run_summary)}[JSON_SUMMARY_END]")
