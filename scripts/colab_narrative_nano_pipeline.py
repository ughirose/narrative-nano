import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Colab GPU Runtime Environment Check ===")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM Allocated:", f"{torch.cuda.memory_allocated(0)/(1024*1024):.2f} MB")

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)
    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))

class NarrativeNanoEncoder(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=256, num_layers=6, num_heads=4, intermediate_dim=1024):
        super().__init__()
        self.embedding = FactorizedEmbedding(vocab_size, 64, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=intermediate_dim, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_modality = nn.Linear(hidden_dim, 2)
        self.head_offset = nn.Linear(hidden_dim, 65)
        self.head_label = nn.Linear(hidden_dim, 8)
        self.head_case = nn.Linear(hidden_dim, 10)
        self.head_epistemic = nn.Linear(hidden_dim, 1)
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(x)
        out_modality = self.head_modality(h)
        out_offset = self.head_offset(h)
        out_label = self.head_label(h)
        out_case = self.head_case(h)
        out_epistemic = torch.sigmoid(self.head_epistemic(h))
        return out_modality, out_offset, out_label, out_case, out_epistemic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NarrativeNanoEncoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"NarrativeNanoEncoder initialized on {device}.")
print(f"Total Parameters: {total_params:,}")

dummy_input = torch.randint(0, 2048, (4, 128), dtype=torch.long).to(device)
outputs = model(dummy_input)
print("Forward Pass Successful! Batch size: 4, Seq len: 128")
print("Modality Output Shape:", list(outputs[0].shape))
print("Offset Output Shape:", list(outputs[1].shape))
print("Label Output Shape:", list(outputs[2].shape))
print("Case Output Shape:", list(outputs[3].shape))
print("Epistemic Output Shape:", list(outputs[4].shape))

model_cpu = model.cpu()
dummy_cpu = dummy_input.cpu()
onnx_path = "/tmp/narrative_nano_5_8m.onnx"
torch.onnx.export(model_cpu, dummy_cpu, onnx_path, input_names=["input_ids"], output_names=["modality", "offset", "label", "case", "epistemic"], dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}}, opset_version=17)
onnx_size = os.path.getsize(onnx_path)
print(f"ONNX Model Exported: {onnx_path} ({onnx_size / (1024*1024):.2f} MB)")
print("Colab Pipeline Ready for Large-Scale Distillation.")
