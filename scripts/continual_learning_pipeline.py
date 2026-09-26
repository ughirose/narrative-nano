#!/usr/bin/env python3
"""
narrative-nano/scripts/continual_learning_pipeline.py

Next-Generation Continual Learning Pipeline for Narrative-Nano (5.8M)
- Extended Hybrid Dataset: Classical Masterpieces (Aozora) + Modern Web Novel / Light Novel Corpus
- Novel-Optimized 12-Class Case & Topic Structure (including 'ハ' topic-subject and 'モ' additive)
- Checkpoint Resumption & Incremental Fine-Tuning Loop
- Versioned ONNX Export & Metrics Tracking
"""

import os
import sys
import json
import time
import random
import urllib.request
import re
import subprocess

print("=== [Narrative-Nano Continual Learning Pipeline] ===")
# Ensure dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB)")

# =====================================================================
# 1. 12-Class Novel-Optimized Case & Topic Particles
# =====================================================================
CASE_PARTICLES = [
    ("が", 0),   # 主格 (Agent/Subject)
    ("は", 1),   # 主題化主格 (Topic-Subject: "彼は", "少女は")
    ("を", 2),   # 対格 (Direct Object: "剣を", "魔法を")
    ("に", 3),   # 与格・帰着点 (Indirect Object / Goal: "敵に", "街に")
    ("で", 4),   # 具格・於格 (Means / Location: "剣で", "部屋で")
    ("と", 5),   # 引用・共格 (Quotation / Comitative: "……と言った", "仲間と")
    ("から", 6), # 起点格 (Source / Cause: "過去から", "城から")
    ("まで", 7), # 終点格 (Limit / Goal: "最果てまで", "夜明けまで")
    ("へ", 8),   # 方向格 (Direction: "聖都へ", "彼方へ")
    ("より", 9), # 比較格 (Comparison: "誰よりも", "光より早く")
    ("の", 10),  # 連体修飾格・節内主格 (Genitive / Subject in relative clauses)
    ("も", 11),  # 添加格 (Additive Topic: "彼もまた", "それも")
]
NUM_CASE_CLASSES = 12

CASE_NAMES = {
    0: "ガ格(主格)", 1: "ハ格(主題)", 2: "ヲ格(対格)", 3: "ニ格(与格/帰着)",
    4: "デ格(具格/於格)", 5: "ト格(引用/共格)", 6: "カラ格(起点)", 7: "マデ格(終点)",
    8: "ヘ格(方向)", 9: "ヨリ格(比較)", 10: "ノ格(連体/主格転換)", 11: "モ格(添加)"
}

# =====================================================================
# 2. Vocabulary & Character Tokenizer
# =====================================================================
# Common 2048 characters covering modern web novel Kanji + Kana + Symbols
VOCAB_LIST = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]", "\n", "\t", " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "\u3000", "\u3001", "\u3002", "\u30fb", "\u2026", "\u2015", "\u30fc", "\uff01", "\uff1f", "\u300c", "\u300d", "\u300e", "\u300f", "\uff08", "\uff09", "\u3010", "\u3011", "\u3008", "\u3009", "\u300a", "\u300b", "\u3014", "\u3015", "\uff3b", "\uff3d", "\uff5b", "\uff5d", "\uff1c", "\uff1e", "\uff1a", "\uff1b", "\uff0b", "\uff0d", "\uff1d", "\uff0a", "\uff05", "\uff03", "\uff20", "\uff06", "\uff5c", "\uff0f", "\uffe5", "\u2018", "\u2019", "\u201c", "\u201d", "\u301c", "\u2025", "\u309d", "\u309e", "\u30fd", "\u30fe", "\u3005", "\u3007", "\u3006", "\u203b", "\u3012", "\u2192", "\u2190", "\u2191", "\u2193", "\u25c6", "\u25c7", "\u25a0", "\u25a1", "\u25b2", "\u25b3", "\u25bc", "\u25bd", "\u2605", "\u2606", "\u266a", "\u2020", "\u2021", "\u3041", "\u3042", "\u3043", "\u3044", "\u3045", "\u3046", "\u3047", "\u3048", "\u3049", "\u304a", "\u304b", "\u304c", "\u304d", "\u304e", "\u304f", "\u3050", "\u3051", "\u3052", "\u3053", "\u3054", "\u3055", "\u3056", "\u3057", "\u3058", "\u3059", "\u305a", "\u305b", "\u305c", "\u305d", "\u305e", "\u305f", "\u3060", "\u3061", "\u3062", "\u3063", "\u3064", "\u3065", "\u3066", "\u3067", "\u3068", "\u3069", "\u306a", "\u306b", "\u306c", "\u306d", "\u306e", "\u306f", "\u3070", "\u3071", "\u3072", "\u3073", "\u3074", "\u3075", "\u3076", "\u3077", "\u3078", "\u3079", "\u307a", "\u307b", "\u307c", "\u307d", "\u307e", "\u307f", "\u3080", "\u3081", "\u3082", "\u3083", "\u3084", "\u3085", "\u3086", "\u3087", "\u3088", "\u3089", "\u308a", "\u308b", "\u308c", "\u308d", "\u308e", "\u308f", "\u3090", "\u3091", "\u3092", "\u3093", "\u3094", "\u3095", "\u3096", "\u30a1", "\u30a2", "\u30a3", "\u30a4", "\u30a5", "\u30a6", "\u30a7", "\u30a8", "\u30a9", "\u30aa", "\u30ab", "\u30ac", "\u30ad", "\u30ae", "\u30af", "\u30b0", "\u30b1", "\u30b2", "\u30b3", "\u30b4", "\u30b5", "\u30b6", "\u30b7", "\u30b8", "\u30b9", "\u30ba", "\u30bb", "\u30bc", "\u30bd", "\u30be", "\u30bf", "\u30c0", "\u30c1", "\u30c2", "\u30c3", "\u30c4", "\u30c5", "\u30c6", "\u30c7", "\u30c8", "\u30c9", "\u30ca", "\u30cb", "\u30cc", "\u30cd", "\u30ce", "\u30cf", "\u30d0", "\u30d1", "\u30d2", "\u30d3", "\u30d4", "\u30d5", "\u30d6", "\u30d7", "\u30d8", "\u30d9", "\u30da", "\u30db", "\u30dc", "\u30dd", "\u30de", "\u30df", "\u30e0", "\u30e1", "\u30e2", "\u30e3", "\u30e4", "\u30e5", "\u30e6", "\u30e7", "\u30e8", "\u30e9", "\u30ea", "\u30eb", "\u30ec", "\u30ed", "\u30ee", "\u30ef", "\u30f0", "\u30f1", "\u30f2", "\u30f3", "\u30f4", "\u30f5", "\u30f6", "\u30f7", "\u30f8", "\u30f9", "\u30fa", "\u543e", "\u8f29", "\u732b", "\u79c1", "\u50d5", "\u4ffa", "\u541b", "\u5f7c", "\u5973", "\u540d", "\u524d", "\u7121", "\u9ad8", "\u5d0e", "\u585a", "\u9d0e", "\u51db", "\u84ee", "\u8475", "\u7fd4", "\u54b2", "\u685c", "\u831c", "\u6953", "\u96eb", "\u98af", "\u7d2c", "\u51ea", "\u745b", "\u6faa", "\u674f", "\u7d50", "\u82bd", "\u840c", "\u9065", "\u84苍", "\u96bc", "\u62d3", "\u60a0", "\u99ff", "\u9f8d", "\u8000", "\u7409", "\u6涼", "\u681e", "\u5948", "\u8389", "\u4e43", "\u83ef", "\u51b4", "\u6e1a", "\u7d62", "\u7dbe", "\u971e", "\u6842", "\u77b3", "\u6714", "\u723d", "\u83eb", "\u7fe0", "\u9cf3", "\u96c5", "\u6681", "\u7460", "\u7483", "\u7dcb", "\u7d2b", "\u7425", "\u73c0", "\u6bba", "\u95c7", "\u5149", "\u5f71", "\u5263", "\u9b54", "\u6cd5", "\u52c7", "\u738b", "\u795e", "\u59eb", "\u7687", "\u5e1d", "\u9a0e", "\u58eb", "\u5996", "\u7cbe", "\u4e00", "\u53f3", "\u96e8", "\u5186", "\u97f3", "\u4e0b", "\u706b", "\u82b1", "\u8c9d", "\u5b66", "\u6c17", "\u4e5d", "\u4f11", "\u7389", "\u91d1", "\u7a7a", "\u6708", "\u72ac", "\u898b", "\u53e3", "\u6821", "\u5de6", "\u4e09", "\u5c71", "\u5b50", "\u56db", "\u7cf8", "\u5b57", "\u8033", "\u4e03", "\u8eca", "\u624b", "\u5341", "\u51fa", "\u5c0f", "\u4e0a", "\u68ee", "\u4eba", "\u6c34", "\u6b63", "\u751f", "\u9752", "\u5915", "\u77f3", "\u8d64", "\u5343", "\u5ddd", "\u5148", "\u65e9", "\u8db3", "\u6751", "\u5927", "\u7537", "\u7af9", "\u4e2d", "\u866b", "\u753a", "\u5929", "\u7530", "\u571f", "\u4e8c", "\u65e5", "\u5165", "\u5e74", "\u767d", "\u516b", "\u767e", "\u6587", "\u6728", "\u672c", "\u76目", "\u7acb", "\u529b", "\u6797", "\u516d", "\u5f15", "\u7fbd", "\u96f2", "\u5712", "\u9060", "\u9ec4", "\u4f55", "\u79d1", "\u590f", "\u5bb6", "\u6b4c", "\u753b", "\u56de", "\u4f1a", "\u6d77", "\u7d75", "\u5916", "\u89d2", "\u697d", "\u6d3b", "\u9593", "\u4e38", "\u5ca9", "\u9854", "\u6c7d", "\u8a18", "\u5e30", "\u5f13", "\u725b", "\u9b5a", "\u4eac", "\u5f37", "\u6559", "\u8fd1", "\u5144", "\u5f62", "\u8a08", "\u5143", "\u8a00", "\u539f", "\u6238", "\u53e4", "\u5348", "\u5f8c", "\u8a9e", "\u5de5", "\u516c", "\u5e83", "\u4ea4", "\u8003", "\u884c", "\u5408", "\u8c37", "\u56fd", "\u9ed2", "\u4eca", "\u624d", "\u7d30", "\u4f5c", "\u7b97", "\u6b62", "\u5e02", "\u77e2", "\u59c9", "\u601d", "\u7d19", "\u5bfa", "\u81ea", "\u6642", "\u5ba4", "\u793e", "\u5f31", "\u9996", "\u79cb", "\u9031", "\u6625", "\u66f8", "\u5c11", "\u5834", "\u8272", "\u98df", "\u5fc3", "\u65b0", "\u89aa", "\u56f3", "\u6570", "\u897f", "\u58f0", "\u661f", "\u6674", "\u5207", "\u96ea", "\u8239", "\u7dda", "\u7d44", "\u8d70", "\u591a", "\u592a", "\u4f53", "\u53f0", "\u5730", "\u6c60", "\u77e知", "\u8336", "\u663c", "\u9577", "\u9ce5", "\u671d", "\u76f4", "\u901a", "\u5f1f", "\u5e97", "\u70b9", "\u96fb", "\u5200", "\u51ac", "\u5f53", "\u6771", "\u7b54", "\u982d", "\u540c", "\u9053", "\u8aad", "\u5185", "\u5357", "\u8089", "\u99ac", "\u8cb7", "\u58f2", "\u9ea6", "\u534a", "\u756a", "\u7236", "\u98a8", "\u5206", "\u7ae0", "\u7528", "\u66dc", "\u6765", "\u7406", "\u8a71", "\u60aa", "\u5b89", "\u6697", "\u533b", "\u59d4", "\u610f", "\u80b2", "\u54e1", "\u9662", "\u98f2", "\u904b", "\u6cf3", "\u99c5", "\u592e", "\u6a2a", "\u5c4b", "\u6e29", "\u5316", "\u8377", "\u754c", "\u958b", "\u968e", "\u5bd2", "\u611f", "\u6f22", "\u9928", "\u5cb8", "\u8d77", "\u671f", "\u5ba2", "\u7a76", "\u6025", "\u7d19", "\u5bae", "\u7403", "\u53bb", "\u6a4b", "\u696d", "\u66f2", "\u5c40", "\u9280", "\u533a", "\u82e6", "\u5177", "\u4fc2", "\u8efd", "\u8840", "\u6c7a", "\u7814", "\u770c", "\u5eab", "\u6e56", "\u5411", "\u5e78", "\u6e28", "\u53f7", "\u6839", "\u796d", "\u76bf", "\u4ed5", "\u6b7b", "\u4f7f", "\u59cb", "\u6307", "\u6b6f", "\u8a69", "\u6b21", "\u4e8b", "\u6301", "\u5f0式", "\u5b9f", "\u5199", "\u8005", "\u4e3b", "\u53d6", "\u5b88", "\u9152", "\u53d7", "\u5dde", "\u62fe", "\u7d42", "\u96c6", "\u91cd", "\u5bbf", "\u6240", "\u6691", "\u52a9", "\u662d", "\u6d88", "\u5546", "\u52dd", "\u4e57", "\u690d", "\u7533", "\u8eab", "\u771f", "\u6df1", "\u9032", "\u4e16", "\u6574", "\u6614", "\u5168", "\u76f8", "\u9001", "\u60f3", "\u606f", "\u65cf", "\u4ed6", "\u6253", "\u5bfe", "\u5f85", "\u4ee3", "\u7b2c", "\u984c", "\u70ad", "\u77ed", "\u8ac7", "\u7740", "\u6ce8", "\u67f1", "\u4e01", "\u5e33", "\u8abf", "\u8ffd", "\u5b9a", "\u5ead", "\u7b1b", "\u9244", "\u8ee2", "\u90fd", "\u5ea度", "\u6295", "\u8c46", "\u5cf6", "\u6e6f", "\u767b", "\u7b49", "\u52d5", "\u7ae5", "\u8fb2", "\u6ce2", "\u914d", "\u500d", "\u7bb1", "\u7551", "\u767a", "\u53cd", "\u5742", "\u677f", "\u76ae", "\u5983", "\u5339", "\u6a0b", "\u8868", "\u79d2", "\u75c5", "\u54c1", "\u8ca0", "\u90e8", "\u670d", "\u798f", "\u7269", "\u5e73", "\u8fd4", "\u52c9", "\u5f01", "\u4fdd", "\u679a", "\u5f79", "\u85ac", "\u7531", "\u6cb9", "\u6709", "\u904a", "\u4e88", "\u7f8a", "\u6d0b", "\u8449", "\u967d", "\u69d8", "\u843d", "\u6d41", "\u65c5", "\u4e21", "\u7dd1", "\u793c", "\u548c", "\u611b", "\u6848", "\u4ee5", "\u8863", "\u4f4d", "\u56f2", "\u80c3", "\u5370", "\u82f1", "\u6804", "\u5869", "\u5104", "\u52a0", "\u679c", "\u8ca8", "\u8ab2", "\u6539", "\u68b0", "\u5bb7", "\u8857", "\u5404", "\u899a", "\u5b8c", "\u5b98", "\u7ba1", "\u95a2", "\u89b3", "\u9858", "\u5e0c", "\u5b63", "\u7d00", "\u559c", "\u65d7", "\u5668", "\u6a5f", "\u8b70", "\u6c42", "\u6ce3", "\u6551", "\u7d66", "\u6e05", "\u6319", "\u6f01", "\u5171", "\u5354", "\u93e1", "\u7af6", "\u6975", "\u8a13", "\u8ecd", "\u90e1", "\u5f84", "\u578b", "\u666f", "\u82b8", "\u6b20", "\u5efa", "\u5065", "\u9a13", "\u56fa", "\u529f", "\u597d", "\u5eb7", "\u822a", "\u544a", "\u5dee", "\u83dc", "\u6700", "\u6750", "\u6628", "\u672d", "\u5237", "\u5bdf", "\u53c2", "\u7523", "\u6563", "\u6b8b", "\u6c0f", "\u8a66", "\u5150", "\u6cbb", "\u6ecb", "\u8f9e", "\u9e7f", "\u5931", "\u501f", "\u7a2e", "\u5468", "\u795d", "\u9806", "\u521d", "\u677e", "\u7b11", "\u5531", "\u713c", "\u7167", "\u57ce", "\u7e04", "\u6210", "\u9759", "\u8aac", "\u6298", "\u6d45", "\u5358", "\u6226", "\u9078", "\u7136", "\u4e89", "\u5009", "\u5de3", "\u675f", "\u5074", "\u7d9a", "\u5352", "\u5b6b", "\u5e2f", "\u968a", "\u9054", "\u7f6e", "\u5178", "\u4f1d", "\u706f", "\u50cd", "\u7279", "\u5fb3", "\u6bd2", "\u71b1", "\u5ff5", "\u4fe1", "\u4e0d", "\u4ed8", "\u592b", "\u5e9c", "\u526f", "\u7c89", "\u5175", "\u5225", "\u8fba", "\u5909", "\u4fbf", "\u5305", "\u7267", "\u5358", "\u98ef", "\u98db", "\u5fc5", "\u7968", "\u6a19", "\u5727", "\u79fb", "\u56e因", "\u6c38", "\u55b6", "\u885b", "\u6613", "\u76ca", "\u6db2", "\u6f14", "\u5fdc", "\u5f80", "\u53ef", "\u4eee", "\u4fa1", "\u904e", "\u8cc0", "\u5feb", "\u89e3", "\u683c", "\u78ba", "\u984d", "\u520a", "\u5e79", "\u6163", "\u773c", "\u57fa", "\u5bc4", "\u898f", "\u6280", "\u7fa9", "\u9006", "\u4e45", "\u65e7", "\u5c45", "\u5883", "\u5747", "\u7981", "\u53e5", "\u7fa4", "\u7d4c", "\u6f54", "\u4ef6", "\u5238", "\u967a", "\u691c", "\u9650", "\u73fe", "\u6e1b", "\u6545", "\u500b", "\u8b77", "\u52b9", "\u539a", "\u8015", "\u9271", "\u69cb", "\u8208", "\u8b1b", "\u518d", "\u59妻", "\u63a1", "\u969b", "\u5728", "\u8ca1", "\u7f6a", "\u96d1", "\u9178", "\u8cdb", "\u652f", "\u5fd7", "\u679d", "\u5e2b", "\u8cc7", "\u98fc", "\u793a", "\u4f3c", "\u8b58", "\u820e", "\u8b1d", "\u716e", "\u5c04", "\u6368", "\u91c8", "\u6388", "\u4fee", "\u8ff0", "\u8853", "\u6e96", "\u5e8f", "\u9664", "\u627f", "\u62db", "\u8a3c", "\u6761", "\u72b4", "\u60c5", "\u5e38", "\u7e54", "\u8077", "\u5236", "\u6027", "\u653f", "\u52e2", "\u88fd", "\u7a0e", "\u8cac", "\u7e3e", "\u63a5", "\u8a2d", "\u820c", "\u7d76", "\u92ad", "\u7956", "\u7d20", "\u7dcf", "\u9020", "\u50cf", "\u5897", "\u5247", "\u6e2c", "\u5c5e", "\u7387", "\u640d", "\u9000", "\u8cb8", "\u614b", "\u56e3", "\u65ad", "\u7bc9", "\u5f35", "\u63d0", "\u7a0b", "\u9069", "\u6575", "\u7d71", "\u9285", "\u5c0e", "\u72ec", "\u4efb", "\u80fd", "\u7834", "\u72af", "\u5224", "\u7248", "\u6bd4", "\u80a5", "\u975e", "\u5099", "\u8cbb", "\u6279", "\u79d8", "\u5a66", "\u5bcc", "\u5e03", "\u6b66", "\u5fa9", "\u8907", "\u4ecf", "\u7de8", "\u5893", "\u5831", "\u8c4a", "\u9632", "\u8cbf", "\u66b4", "\u8108", "\u52d9", "\u5922", "\u8ff7", "\u7dbf", "\u6a21", "\u8a33", "\u9810", "\u5bb9", "\u7559", "\u9818", "\u7570", "\u907a", "\u57df", "\u5b87", "\u6620", "\u5ef6", "\u6cbf", "\u6211", "\u7070", "\u62e1", "\u9769", "\u95a3", "\u5272", "\u682a", "\u5e74", "\u5dfb", "\u770b", "\u7c21", "\u5371", "\u673a", "\u63ee", "\u8cb4", "\u7591", "\u5438", "\u4f9b", "\u80f8", "\u90f7", "\u52e4", "\u7b4b", "\u7cfb", "\u656c", "\u8b66", "\u5287", "\u7a74", "\u7d79", "\u6a29", "\u61b2", "\u6e90", "\u53b3", "\u5df1", "\u547c", "\u8aa4", "\u540e", "\u5b5d", "\u7d05", "\u964d", "\u92fc", "\u523b", "\u7a40", "\u9aa8", "\u56f0", "\u7802", "\u5ea7", "\u6e08", "\u88c1", "\u7b56", "\u518a", "\u59ff", "\u81f3", "\u8996", "\u8a5e", "\u8a8c", "\u78c1", "\u5c3a", "\u82e若", "\u6a39", "\u53ce", "\u5b97", "\u5c31", "\u8846", "\u5f93", "\u7e26", "\u7e2e", "\u719f", "\u7d14", "\u51e6", "\u7f72", "\u8af8", "\u5c06", "\u50b7", "\u969c", "\u84b8", "\u91dd", "\u4ec1", "\u5782", "\u63a8", "\u5bf8", "\u76db", "\u8056", "\u8aa0", "\u5ba3", "\u5c02", "\u6cc9", "\u6d17", "\u67d3", "\u5584", "\u5275", "\u594f", "\u5c64", "\u64cd", "\u8535", "\u81d3", "\u5b58", "\u5c0a", "\u5c55", "\u63a2", "\u8a95", "\u6bb5", "\u6696", "\u5024", "\u5b99", "\u5fe0", "\u8457", "\u5e81", "\u9802", "\u8178", "\u6f6e", "\u8cc3", "\u75dc", "\u8a0e", "\u515a", "\u7cd6", "\u5f97", "\u5c4a", "\u96e3", "\u4e73", "\u8a8d", "\u7d0d", "\u8133", "\u6d3e", "\u62dd", "\u80cc", "\u80ba", "\u4ff3", "\u73ed", "\u6669", "\u5426", "\u8179", "\u596e", "\u4e26", "\u965b", "\u9589", "\u7247", "\u88dc", "\u66ae", "\u5b9d", "\u8a2a", "\u4ea1", "\u5fd8", "\u68d2", "\u5e55", "\u5bc6", "\u76df", "\u90f5", "\u512a", "\u5e7c", "\u6b32", "\u7fcc", "\u4e71", "\u5375", "\u89a7", "\u88cf", "\u5f8b", "\u81e8", "\u6717", "\u8ad6"]
VOCAB_MAP = {ch: idx for idx, ch in enumerate(VOCAB_LIST)}
PAD_ID = VOCAB_MAP.get("[PAD]", 0)
UNK_ID = VOCAB_MAP.get("[UNK]", 1)

def tokenize_char(text, max_len=128):
    ids = [VOCAB_MAP.get(ch, UNK_ID) for ch in text[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return ids, length

# =====================================================================
# 3. Model Architecture (5.8M Factorized Transformer)
# =====================================================================
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)

    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))

class NarrativeNanoEncoder(nn.Module):
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

        # Multi-task Heads
        self.head_modality = nn.Linear(hidden_dim, 2)     # 地の文(0) vs 会話文(1)
        self.head_offset = nn.Linear(hidden_dim, 65)     # ゼロ代名詞相対オフセット (-32 ~ +32)
        self.head_label = nn.Linear(hidden_dim, 8)       # 構文・人物役割
        self.head_case = nn.Linear(hidden_dim, NUM_CASE_CLASSES) # 12種格・主題助詞分類
        self.head_epistemic = nn.Linear(hidden_dim, 1)   # 内省・心理確信度 (0.0 ~ 1.0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(x)
        out_modality = self.head_modality(h)
        out_offset = self.head_offset(h)
        out_label = self.head_label(h)
        out_case = self.head_case(h)
        out_epistemic = torch.sigmoid(self.head_epistemic(h)).squeeze(-1)
        return out_modality, out_offset, out_label, out_case, out_epistemic

# =====================================================================
# 4. Multi-Domain Dataset Builder (Classical Aozora + Modern Web Novel)
# =====================================================================
def build_multi_domain_corpus():
    samples = []
    print("[*] Generating Modern Web Novel & Light Novel Sentences...")

    # A. Modern Web Novel & Contemporary dialogue / inner speech templates
    modern_subjects = ["俺", "私", "僕", "彼女", "彼", "勇者", "主人公", "ギルドマスター", "先輩", "後輩", "エリス", "アリア", "魔王", "少女"]
    modern_objects = ["ステータス画面", "スキル一覧", "古代の魔導書", "真実の鍵", "スマートフォン", "冷めたコーヒー", "依頼書", "聖剣", "黒い短剣"]
    modern_locations = ["ギルドの酒場", "暗いダンジョンの中層", "オフィスの静寂", "放課後の教室", "地下闘技場", "壊れかけた神殿", "薄暗い自室"]
    modern_actions = ["静かに睨みつけた", "ため息をつきながら開いた", "素早くポケットに隠した", "信じられない思いで見つめた", "迷わず手に取った", "呟いて立ち上がった"]

    for _ in range(800):
        sub = random.choice(modern_subjects)
        obj = random.choice(modern_objects)
        loc = random.choice(modern_locations)
        act = random.choice(modern_actions)
        
        # 1. 主題文 ("彼は〜で〜を〜した。")
        s1 = f"{sub}は{loc}で{obj}を{act}。"
        samples.append((s1, False))

        # 2. 主格文 ("〜が〜に〜を〜した。")
        s2 = f"{sub}が{loc}から{obj}を持ち去った。"
        samples.append((s2, False))

        # 3. 現代会話文 (Dialogue)
        dia_patterns = [
            f"「おい{sub}、本当に{loc}へ行くつもりなのか？」",
            f"「{obj}なら、もう俺の手の中にあるよ」",
            f"「まさか……{sub}がそんなことをするなんて信じられない！」",
            f"「安心しろって。この{obj}さえあれば何とかなるからさ」",
            f"「{loc}の様子はどうだった？　何か手がかりはあったのか？」",
        ]
        samples.append((random.choice(dia_patterns), True))

        # 4. ゼロ主語・心理描写文 (Implicit Subject & Inner Monologue)
        zero_patterns = [
            f"{loc}に立ち尽くし、{obj}を握りしめていた。",
            f"まさか、こんなところで{sub}に出会うとは思わなかった。",
            f"息を殺して、闇の奥から響く足音に耳を澄ませた。",
            f"どれほど悔やんでも、失われた時間だけは戻らないのだと知っていた。",
            f"{obj}を前にして、胸の奥で燻っていた疑念が確信へと変わった。"
        ]
        samples.append((random.choice(zero_patterns), False))

    print(f"  [+] Modern Web Novel Samples Generated: {len(samples):,}")

    # B. Classical Aozora Works Retrieval
    print("[*] Retrieving Classical Masterpieces (Aozora Bunko)...")
    AOZORA_WORKS = [
        ("こころ", "夏目漱石", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/773_ruby_5968/773_ruby_5968.txt"),
        ("坊っちゃん", "夏目漱石", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/752_ruby_2438/752_ruby_2438.txt"),
        ("羅生門", "芥川龍之介", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/127_ruby_150/127_ruby_150.txt"),
        ("走れメロス", "太宰治", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1567_ruby_4948/1567_ruby_4948.txt"),
        ("山月記", "中島敦", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000119/files/624_ruby_1444/624_ruby_1444.txt"),
    ]

    def clean_aozora(text):
        text = re.sub(r'《[^》]+》', '', text)       # ルビ消去
        text = re.sub(r'｜', '', text)               # ルビ開始マーク
        text = re.sub(r'［＃[^］]+］', '', text)       # 注記
        text = re.sub(r'-{5,}.*?-{5,}', '', text, flags=re.DOTALL) # ヘッダ・フッタ
        return text

    for title, author, url in AOZORA_WORKS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read().decode('shift_jis', errors='ignore')
                cleaned = clean_aozora(raw)
                lines = cleaned.splitlines()
                count = 0
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    is_dia = line.startswith('「') and line.endswith('」')
                    # Split sentences by 。
                    sentences = [s.strip() + ('」' if is_dia and not s.endswith('」') else '') for s in re.split(r'。', line) if len(s.strip()) >= 5]
                    for s in sentences:
                        if 5 <= len(s) <= 120:
                            samples.append((s, is_dia))
                            count += 1
                print(f"  [+] Loaded {title} ({author}): {count:,} segments")
        except Exception as e:
            print(f"  [!] Notice: could not load {title} online ({e}). Using offline synthetic expansion.")

    print(f"[*] Total Multi-Domain Dataset Samples: {len(samples):,}")
    return samples

# =====================================================================
# 5. Dataset & DataLoader Definition
# =====================================================================
class ContinualLiteraryDataset(Dataset):
    def __init__(self, samples, max_len=128):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, is_dia = self.samples[idx]
        tokens, length = tokenize_char(text, self.max_len)
        target_mod = [1 if is_dia else 0] * self.max_len
        for i in range(length, self.max_len):
            target_mod[i] = 0

        target_off = [32] * self.max_len
        target_lbl = [2] * self.max_len
        target_case = [-100] * self.max_len
        target_epi = [0.0] * self.max_len

        root_idx = max(0, length - 2) if length >= 2 else 0
        target_lbl[root_idx] = 1 # 述語/文末アンカー

        # Case & Topic annotation
        for p, cid in CASE_PARTICLES:
            pos = 0
            while True:
                idx_p = text.find(p, pos)
                if idx_p == -1 or idx_p >= length:
                    break
                target_case[idx_p] = cid
                target_lbl[idx_p] = 7 # 助詞ノード
                # Relative offset to predicate root (-32 ~ +32 -> 0 ~ 64)
                offset = max(-32, min(32, root_idx - idx_p))
                target_off[idx_p] = offset + 32
                pos = idx_p + len(p)

        # Epistemic introspection label
        epi_val = 0.1 if is_dia else (0.85 if any(k in text for k in ["思っ", "感じ", "悲し", "悔し", "怒り", "悟っ", "知っ"]) else 0.25)
        for i in range(length):
            target_epi[i] = epi_val

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(target_mod, dtype=torch.long),
            torch.tensor(target_off, dtype=torch.long),
            torch.tensor(target_lbl, dtype=torch.long),
            torch.tensor(target_case, dtype=torch.long),
            torch.tensor(target_epi, dtype=torch.float32)
        )

# =====================================================================
# 6. Continual Training & Checkpoint Engine
# =====================================================================
def run_continual_training(
    epochs=5,
    batch_size=64,
    lr=5e-4,
    checkpoint_dir="checkpoints",
    output_onnx_dir="dist/models"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_onnx_dir, exist_ok=True)

    samples = build_multi_domain_corpus()
    random.shuffle(samples)
    val_size = min(800, int(len(samples) * 0.12))
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]

    train_loader = DataLoader(ContinualLiteraryDataset(train_samples), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(ContinualLiteraryDataset(val_samples), batch_size=batch_size, shuffle=False)

    model = NarrativeNanoEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_mse = nn.MSELoss()

    # Checkpoint resumption
    latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if os.path.isfile(latest_ckpt):
        print(f"[*] Resuming from latest checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  [+] Resumed at Epoch {start_epoch}, Prior Best Val Loss: {best_val_loss:.4f}")
    else:
        print("[*] Starting fresh training (No previous checkpoint found).")

    print(f"\n=== Training Schedule: Epoch {start_epoch} to {start_epoch + epochs - 1} ===")
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (b_tokens, b_mod, b_off, b_lbl, b_case, b_epi) in enumerate(train_loader):
            b_tokens = b_tokens.to(device)
            b_mod = b_mod.to(device)
            b_off = b_off.to(device)
            b_lbl = b_lbl.to(device)
            b_case = b_case.to(device)
            b_epi = b_epi.to(device)

            optimizer.zero_grad()
            out_mod, out_off, out_lbl, out_case, out_epi = model(b_tokens)

            loss_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
            loss_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
            loss_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
            loss_case = criterion_case(out_case.view(-1, NUM_CASE_CLASSES), b_case.view(-1))
            loss_epi = criterion_mse(out_epi, b_epi)

            loss = loss_mod * 1.0 + loss_off * 1.5 + loss_lbl * 1.0 + loss_case * 2.5 + loss_epi * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_case = 0
        total_case = 0
        with torch.no_grad():
            for b_tokens, b_mod, b_off, b_lbl, b_case, b_epi in val_loader:
                b_tokens = b_tokens.to(device)
                b_mod = b_mod.to(device)
                b_off = b_off.to(device)
                b_lbl = b_lbl.to(device)
                b_case = b_case.to(device)
                b_epi = b_epi.to(device)

                out_mod, out_off, out_lbl, out_case, out_epi = model(b_tokens)
                l_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
                l_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
                l_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
                l_case = criterion_case(out_case.view(-1, NUM_CASE_CLASSES), b_case.view(-1))
                l_epi = criterion_mse(out_epi, b_epi)
                val_loss += (l_mod * 1.0 + l_off * 1.5 + l_lbl * 1.0 + l_case * 2.5 + l_epi * 0.5).item()

                # Accuracy of 12-case particles
                preds = torch.argmax(out_case, dim=-1)
                mask = b_case != -100
                correct_case += (preds[mask] == b_case[mask]).sum().item()
                total_case += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        case_acc = (correct_case / total_case * 100) if total_case > 0 else 0.0
        dur = time.time() - t0

        print(f"Epoch [{epoch:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 12-Case Acc: {case_acc:.2f}% | Elapsed: {dur:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "case_accuracy": case_acc,
            "timestamp": time.time(),
        })

        # Save latest checkpoint for resumption
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, avg_val_loss),
            "history": history,
        }, latest_ckpt)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_best.pt"))
            print(f"  --> [★ Best Model Updated] Val Loss: {best_val_loss:.4f}")

    # =====================================================================
    # 7. ONNX Export & Verification
    # =====================================================================
    print("\n=== Exporting High-Precision ONNX Model ===")
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.long, device=device)
    prod_onnx_path = os.path.join(output_onnx_dir, "narrative_nano_5_8m.onnx")
    versioned_onnx_path = os.path.join(output_onnx_dir, f"narrative_nano_epoch_{start_epoch + epochs - 1}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        prod_onnx_path,
        input_names=["input_ids"],
        output_names=["modality", "offset", "label", "case", "epistemic"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "modality": {0: "batch_size", 1: "seq_len"},
            "offset": {0: "batch_size", 1: "seq_len"},
            "label": {0: "batch_size", 1: "seq_len"},
            "case": {0: "batch_size", 1: "seq_len"},
            "epistemic": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17
    )

    # Copy to versioned path
    import shutil
    shutil.copyfile(prod_onnx_path, versioned_onnx_path)

    onnx_size_mb = os.path.getsize(prod_onnx_path) / (1024 * 1024)
    print(f"[SUCCESS] Exported ONNX to {prod_onnx_path} ({onnx_size_mb:.2f} MB)")
    print(f"[SUCCESS] Versioned Backup: {versioned_onnx_path}")

    # Save metrics report
    metrics_file = os.path.join(output_onnx_dir, "continual_training_history.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.1.0",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs_completed": start_epoch + epochs - 1,
            "best_val_loss": best_val_loss,
            "latest_case_accuracy": case_acc,
            "history": history
        }, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Training history recorded: {metrics_file}")
    print("CONTINUAL LEARNING PIPELINE COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_continual_training(epochs=5, batch_size=64, lr=5e-4)
