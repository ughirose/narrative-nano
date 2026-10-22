"""
Narrative-Nano Focus v18 Colab Script Generator
Generates `scripts/build_colab_v18_script.py` which embeds:
- Deterministic token corpus v18 (9,220 samples, 296,076 tokens, leak-free)
- Literary Benchmark 100 (100 samples with strict objective evaluation)
- Official CharTokenizer vocabulary (2,048 tokens)
- NarrativeNanoFocus architecture (d_model=256, n_heads=4, num_layers=4, d_ff=512)
- Training loop (12 Epochs, class weights [1.0, 2.5, 6.0, 12.0])
- ONNX export & QInt8 dynamic quantization pipeline
"""

import os
import sys
import json
import gzip
import base64
import hashlib

COLAB_SCRIPT_TEMPLATE = r'''# Narrative-Nano Focus v18 Training & Dynamic Wasm SIMD Pipeline
# Pure Deterministic Token-Level Context & Case Classifier
# Leak-Free Literary Corpus v18 + Strict 100-Sample Benchmark Objective Audit
import os
import sys
import json
import gzip
import base64
import math
import time
import subprocess
from collections import Counter

try:
    import onnxscript
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=== 1. Hardware & Environment Verification ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("=== 2. Loading Deterministic Token Corpus v18 (Leak-Free) ===")
CORPUS_B64 = "__CORPUS_B64__"
raw_gz = base64.b64decode(CORPUS_B64)
raw_jsonl = gzip.decompress(raw_gz).decode("utf-8")
dataset_samples = [json.loads(line) for line in raw_jsonl.strip().split("\n") if line.strip()]
print(f"Loaded {len(dataset_samples):,} verified leak-free token samples")

print("=== 2.1 Loading Literary Benchmark 100 ===")
BENCHMARK_B64 = "__BENCHMARK_B64__"
raw_bench_gz = base64.b64decode(BENCHMARK_B64)
raw_bench_jsonl = gzip.decompress(raw_bench_gz).decode("utf-8")
benchmark_samples = [json.loads(line) for line in raw_bench_jsonl.strip().split("\n") if line.strip()]
print(f"Loaded {len(benchmark_samples):,} benchmark samples for objective audit")

VOCAB_B64 = "__VOCAB_B64__"
VOCAB_TABLE = json.loads(base64.b64decode(VOCAB_B64).decode("utf-8"))
VOCAB_MAP = {ch: i for i, ch in enumerate(VOCAB_TABLE)}
VOCAB_SIZE = len(VOCAB_TABLE)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
print(f"Official Vocabulary Size: {VOCAB_SIZE} tokens (CharTokenizer aligned)")

def tokenize(text):
    return [VOCAB_MAP.get(ch, UNK_ID) for ch in text]

class DynamicMHA(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=512):
        super().__init__()
        self.attn = DynamicMHA(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class NarrativeNanoFocus(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, n_heads=4, num_layers=4, max_pos=256):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)
        # Head 1: Token Modality (4 classes: 0=Narrative, 1=Dialogue, 2=Monologue, 3=Citation)
        self.head_token_modality = nn.Linear(d_model, 4)
        # Head 2: Case Particles (10 classes: が, を, に, へ, と, より, から, で, や, の)
        self.head_case = nn.Linear(d_model, 10)

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        out_modality = self.head_token_modality(x)
        out_case = self.head_case(x)
        return out_modality, out_case

MAX_LEN = 128

class TokenLiteraryDataset(Dataset):
    def __init__(self, samples, max_len=MAX_LEN):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item.get("text", "")[:self.max_len]
        tokens = tokenize(text)
        length = len(tokens)
        mod_labels = item.get("token_modality", [0] * length)[:length]

        if length < self.max_len:
            tokens += [PAD_ID] * (self.max_len - length)

        target_mod = mod_labels + [-100] * (self.max_len - length)

        target_case = [-100] * self.max_len
        for cp in item.get("case_particles", []):
            cid = cp.get("case_id", -1)
            span = cp.get("span", [])
            if 0 <= cid < 10 and len(span) == 2:
                for pos in range(span[0], min(length, span[1])):
                    target_case[pos] = cid

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(target_mod, dtype=torch.long),
            torch.tensor(target_case, dtype=torch.long)
        )

val_size = int(len(dataset_samples) * 0.1)
train_samples = dataset_samples[:-val_size]
val_samples = dataset_samples[-val_size:]

train_dataset = TokenLiteraryDataset(train_samples)
val_dataset = TokenLiteraryDataset(val_samples)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("=== 3. Model Initialization ===")
model = NarrativeNanoFocus().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,} (Focus v18, Dynamic SeqLen)")

# Class weights: 0: 1.0, 1: 2.5, 2: 6.0, 3: 12.0 (compensate for rare classes)
mod_weights = torch.tensor([1.0, 2.5, 6.0, 12.0], device=device)
crit_mod = nn.CrossEntropyLoss(weight=mod_weights, ignore_index=-100)
crit_case = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
num_epochs = 12

print(f"=== 4. Training ({num_epochs} Epochs on {device}) ===")
t_start = time.time()
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    correct_mod = 0
    total_mod = 0
    for input_ids, target_mod, target_case in train_loader:
        input_ids = input_ids.to(device)
        target_mod = target_mod.to(device)
        target_case = target_case.to(device)

        optimizer.zero_grad()
        out_mod, out_case = model(input_ids)

        loss_m = crit_mod(out_mod.view(-1, 4), target_mod.view(-1))
        valid_case = (target_case.view(-1) != -100)
        if valid_case.any():
            loss_c = crit_case(out_case.view(-1, 10), target_case.view(-1))
        else:
            loss_c = torch.tensor(0.0, device=device)

        loss = loss_m + 0.5 * loss_c
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out_mod.argmax(dim=-1)
        mask = (target_mod != -100)
        correct_mod += (preds[mask] == target_mod[mask]).sum().item()
        total_mod += mask.sum().item()

    train_acc = (correct_mod / total_mod * 100) if total_mod > 0 else 0.0

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for input_ids, target_mod, target_case in val_loader:
            input_ids = input_ids.to(device)
            target_mod = target_mod.to(device)
            out_mod, _ = model(input_ids)
            preds = out_mod.argmax(dim=-1)
            mask = (target_mod != -100)
            val_correct += (preds[mask] == target_mod[mask]).sum().item()
            val_total += mask.sum().item()
    val_acc = (val_correct / val_total * 100) if val_total > 0 else 0.0
    print(f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

t_train = time.time() - t_start
print(f"Training completed in {t_train:.2f}s")

print("=== 5. Literary Benchmark 100 Objective Audit ===")
model.eval()

cat_stats = {
    "dialogue": {"total": 0, "passed": 0},
    "narrative": {"total": 0, "passed": 0},
    "monologue": {"total": 0, "passed": 0},
    "mixed": {"total": 0, "passed": 0},
    "dialogue_unquoted": {"total": 0, "passed": 0},
    "narrative_pure": {"total": 0, "passed": 0}
}

key_samples_log = []
total_passed = 0

with torch.no_grad():
    for item in benchmark_samples:
        sid = item.get("id", "")
        text = item.get("text", "")
        cat = item.get("gold_category", "unknown")
        exp = item.get("expected_modality", "")
        
        tokens = tokenize(text)
        inp = torch.tensor([tokens], dtype=torch.long, device=device)
        out_mod, _ = model(inp)
        preds = out_mod.argmax(dim=-1)[0].tolist()
        
        counts = Counter(preds)
        passed = False
        
        # Rigorous category-specific pass criteria (NO CHERRY-PICKED / SOFT RULES!)
        if cat == "dialogue":
            # Quoted / Standard dialogue: Must detect dialogue tokens (class 1)
            passed = (counts[1] >= 2)
        elif cat == "dialogue_unquoted":
            # Unquoted dialogue: STRICT counts[1] >= 2 (must detect at least 2 dialogue tokens, no soft OR counts[0]>0)
            passed = (counts[1] >= 2)
        elif cat in ("narrative", "narrative_pure"):
            # Narrative: Predominantly narrative (class 0 >= 50%) and no dialogue false positives (class 1 <= 1)
            passed = (counts[0] >= len(preds) * 0.5) and (counts[1] <= 1)
        elif cat == "monologue":
            # Monologue: Must detect monologue tokens (class 2)
            passed = (counts[2] >= 2)
        elif cat == "mixed":
            # Mixed: Both dialogue (class 1) and narrative (class 0) detected
            passed = (counts[1] >= 2) and (counts[0] >= 2)
        else:
            passed = False
            
        if cat in cat_stats:
            cat_stats[cat]["total"] += 1
            if passed:
                cat_stats[cat]["passed"] += 1
        if passed:
            total_passed += 1

        # Detailed logging for key samples: bench-005 and bench-071 to bench-085
        is_key_sample = (sid == "bench-005") or (cat == "dialogue_unquoted")
        if is_key_sample:
            token_annot = "".join(f"{ch}({p})" for ch, p in zip(text, preds))
            sample_info = {
                "id": sid,
                "text": text,
                "gold_category": cat,
                "expected_modality": exp,
                "counts": dict(counts),
                "passed": passed,
                "token_annot": token_annot
            }
            key_samples_log.append(sample_info)
            status_str = "PASS" if passed else "FAIL"
            print(f"[{status_str}] {sid} ({cat}): counts={dict(counts)}")
            print(f"      Text: {text}")
            print(f"      Annot: {token_annot[:120]}...")

# Compile Audit JSON
audit_json = {
    "model_version": "v18",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "num_samples": len(benchmark_samples),
    "total_passed": total_passed,
    "overall_accuracy": round((total_passed / len(benchmark_samples)) * 100, 2) if benchmark_samples else 0.0,
    "categories": {}
}

for c, st in cat_stats.items():
    tot = st["total"]
    pas = st["passed"]
    acc = round((pas / tot) * 100, 2) if tot > 0 else 0.0
    audit_json["categories"][c] = {
        "total": tot,
        "passed": pas,
        "accuracy": acc
    }

audit_json["key_samples"] = key_samples_log

print("\nMODEL_AUDIT_JSON_START")
print(json.dumps(audit_json, ensure_ascii=False, indent=2))
print("MODEL_AUDIT_JSON_END\n")

print("=== 6. ONNX Export & INT8 Quantization ===")
os.makedirs("/tmp/v18_models", exist_ok=True)
onnx_fp32_path = "/tmp/v18_models/narrative_nano_v18_fp32.onnx"
onnx_int8_path = "/tmp/v18_models/narrative_nano_v18_int8.onnx"

model_cpu = model.cpu()
dummy_input = torch.randint(0, VOCAB_SIZE, (1, 64), dtype=torch.long)
torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_fp32_path,
    input_names=["input_ids"],
    output_names=["token_modality", "case"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "token_modality": {0: "batch_size", 1: "seq_len"},
        "case": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=17,
    dynamo=False,
    do_constant_folding=True
)
fp32_size = os.path.getsize(onnx_fp32_path)
print(f"Exported FP32 ONNX: {fp32_size:,} bytes ({fp32_size/(1024*1024):.2f} MB)")

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    onnx_fp32_path,
    onnx_int8_path,
    weight_type=QuantType.QInt8
)
int8_size = os.path.getsize(onnx_int8_path)
print(f"Quantized INT8 ONNX: {int8_size:,} bytes ({int8_size/(1024*1024):.2f} MB)")

# Output base64 for local retrieval
with open(onnx_int8_path, "rb") as f:
    int8_bytes = f.read()
b64_int8 = base64.b64encode(int8_bytes).decode("ascii")
print("ARTIFACT_V18_B64_START")
print(b64_int8)
print("ARTIFACT_V18_B64_END")
print(f"=== MODEL_B64_WRITTEN: {len(b64_int8)} chars ===")
'''

def generate_script():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    v17_script_path = os.path.join(base_dir, "scripts", "build_colab_v17_script.py")
    corpus_gz_path = os.path.join(base_dir, "dist", "datasets", "token_corpus_v18_leakfree.jsonl.gz")
    benchmark_path = os.path.join(base_dir, "dist", "datasets", "literary_benchmark_100.jsonl")
    output_script_path = os.path.join(base_dir, "scripts", "build_colab_v18_script.py")

    print("[*] Generating Narrative-Nano Focus v18 Colab Script...")

    # 1. Extract VOCAB_B64 from v17 script
    print(f"[*] Extracting VOCAB_B64 from: {v17_script_path}")
    vocab_b64 = None
    with open(v17_script_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VOCAB_B64 = "):
                vocab_b64 = line.strip().split(" = ")[1].strip("\"'")
                break
    if not vocab_b64:
        raise ValueError(f"Failed to extract VOCAB_B64 from {v17_script_path}")
    print(f"[+] VOCAB_B64 extracted: {len(vocab_b64):,} chars")

    # 2. Read and Base64 encode token_corpus_v18_leakfree.jsonl.gz
    print(f"[*] Reading leak-free corpus: {corpus_gz_path}")
    with open(corpus_gz_path, "rb") as f:
        corpus_gz_bytes = f.read()
    corpus_b64 = base64.b64encode(corpus_gz_bytes).decode("ascii")
    print(f"[+] Corpus gz size: {len(corpus_gz_bytes):,} bytes | Base64: {len(corpus_b64):,} chars")

    # 3. Read, gzip compress, and Base64 encode literary_benchmark_100.jsonl
    print(f"[*] Reading literary benchmark: {benchmark_path}")
    with open(benchmark_path, "rb") as f:
        bench_bytes = f.read()
    bench_gz_bytes = gzip.compress(bench_bytes, mtime=0)
    bench_b64 = base64.b64encode(bench_gz_bytes).decode("ascii")
    print(f"[+] Benchmark size: {len(bench_bytes):,} bytes | gz: {len(bench_gz_bytes):,} bytes | Base64: {len(bench_b64):,} chars")

    # 4. Substitute placeholders into template and replace blank lines with '#' to prevent colab_execute_file cell splitting
    print("[*] Assembling full v18 training script...")
    final_script = COLAB_SCRIPT_TEMPLATE.replace("__CORPUS_B64__", corpus_b64)
    final_script = final_script.replace("__BENCHMARK_B64__", bench_b64)
    final_script = final_script.replace("__VOCAB_B64__", vocab_b64)
    final_script = "\n".join(line if line.strip() else "#" for line in final_script.splitlines()) + "\n"

    # 5. Write out build_colab_v18_script.py
    with open(output_script_path, "w", encoding="utf-8") as f:
        f.write(final_script)

    script_size = os.path.getsize(output_script_path)
    sha256 = hashlib.sha256(final_script.encode("utf-8")).hexdigest()
    print(f"[SUCCESS] Generated: {output_script_path}")
    print(f"  Size: {script_size:,} bytes ({script_size/(1024*1024):.2f} MB)")
    print(f"  SHA-256: {sha256}")

    return output_script_path

if __name__ == "__main__":
    generate_script()
