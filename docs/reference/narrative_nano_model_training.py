"""
NarrativeNano: Character-level Transformer Encoder with Dynamic Biaffine PAS Head
PyTorch implementation for lightweight narrative syntax and predicate-argument structure parsing.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TARGET_CASES = ["ガ", "ガ２", "ヲ", "ニ", "ト", "デ", "カラ", "ヨリ", "ヘ", "マデ"]

class NarrativeNanoConfig:
    def __init__(self, vocab_size: int = 1500):
        self.vocab_size = vocab_size
        self.hidden_dim = 256
        self.num_layers = 6
        self.num_heads = 4
        self.ffn_dim = 1024
        self.max_seq_len = 128
        self.num_offset_classes = 65  # -32 to +32
        self.num_dep_labels = 8
        self.num_cases = 10
        self.dropout = 0.1

class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = (attention_mask == 0).view(B, 1, 1, L)
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        h = torch.matmul(attn_weights, v)
        h = h.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(h)

class DynamicTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = DynamicMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class DynamicBiaffinePASHead(nn.Module):
    def __init__(self, in_dim: int, num_cases: int):
        super().__init__()
        self.num_cases = num_cases
        self.w_pred = nn.Linear(in_dim, in_dim)
        self.w_arg = nn.Linear(in_dim, in_dim)
        self.u_cases = nn.Parameter(torch.Tensor(num_cases, in_dim, in_dim))
        nn.init.xavier_uniform_(self.u_cases)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        h_pred = self.w_pred(h)
        h_arg = self.w_arg(h)

        h_pred_exp = h_pred.unsqueeze(1)
        h_pred_u = torch.matmul(h_pred_exp, self.u_cases)  # (B, C, L, D)
        scores = torch.matmul(h_pred_u, h_arg.transpose(1, 2).unsqueeze(1))  # (B, C, L, L)
        return scores

class NarrativeNanoRobust(nn.Module):
    def __init__(self, config: NarrativeNanoConfig):
        super().__init__()
        self.config = config
        self.char_embed = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.ln_in = nn.LayerNorm(config.hidden_dim)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DynamicTransformerBlock(config.hidden_dim, config.num_heads, config.ffn_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.modality_head = nn.Linear(config.hidden_dim, 2)
        self.offset_head = nn.Linear(config.hidden_dim, config.num_offset_classes)
        self.label_head = nn.Linear(config.hidden_dim, config.num_dep_labels)
        self.pas_head = DynamicBiaffinePASHead(config.hidden_dim, config.num_cases)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.char_embed(input_ids) + self.pos_embed(positions)
        x = self.drop(self.ln_in(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        logits_modality = self.modality_head(x)
        logits_offset = self.offset_head(x)
        logits_label = self.label_head(x)
        logits_pas = self.pas_head(x)

        return {
            "modality": logits_modality,
            "offset": logits_offset,
            "label": logits_label,
            "pas": logits_pas
        }

class NaturalSpanAggregatorFixed:
    def __init__(self, text: str, dependencies: list, events: list):
        self.text = text
        self.dependencies = dependencies
        self.events = events

    def _collect_subtree(self, head_idx: int, visited: set = None) -> set:
        if visited is None:
            visited = set()
        visited.add(head_idx)
        for dep in self.dependencies:
            child = dep.get("token_idx")
            parent = dep.get("head_idx")
            if parent == head_idx and child not in visited:
                self._collect_subtree(child, visited)
        return visited

    def aggregate(self, conf_th: float = 0.5) -> list:
        results = []
        for ev in self.events:
            p_idx = ev["predicate_idx"]
            cases = ev.get("cases", {})
            args_extracted = {}
            for case_name, arg_indices in cases.items():
                spans = []
                for a_idx in arg_indices:
                    sub_indices = sorted(list(self._collect_subtree(a_idx)))
                    span_text = "".join(self.text[i] for i in sub_indices if i < len(self.text))
                    spans.append(span_text)
                if spans:
                    args_extracted[case_name] = spans
            results.append({
                "predicate": self.text[p_idx:p_idx+4],
                "arguments": args_extracted
            })
        return results
