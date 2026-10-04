#!/usr/bin/env python3
# Colab Narrative-Nano Distillation & Training Pipeline
import os
import sys
import json
import gzip
import base64
import time
import urllib.request
import re
import subprocess
#
print("=== 1. Colab Runtime Environment Setup ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
#
VOCAB_LIST = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[CLS]", "[SEP]", "[RESERVED]", "\n", "\t", " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "\u3000", "\u3001", "\u3002", "\u30fb", "\u2026", "\u2015", "\u30fc", "\uff01", "\uff1f", "\u300c", "\u300d", "\u300e", "\u300f", "\uff08", "\uff09", "\u3010", "\u3011", "\u3008", "\u3009", "\u300a", "\u300b", "\u3014", "\u3015", "\uff3b", "\uff3d", "\uff5b", "\uff5d", "\uff1c", "\uff1e", "\uff1a", "\uff1b", "\uff0b", "\uff0d", "\uff1d", "\uff0a", "\uff05", "\uff03", "\uff20", "\uff06", "\uff5c", "\uff0f", "\uffe5", "\u2018", "\u2019", "\u201c", "\u201d", "\u301c", "\u2025", "\u309d", "\u309e", "\u30fd", "\u30fe", "\u3005", "\u3007", "\u3006", "\u203b", "\u3012", "\u2192", "\u2190", "\u2191", "\u2193", "\u25c6", "\u25c7", "\u25a0", "\u25a1", "\u25b2", "\u25b3", "\u25bc", "\u25bd", "\u2605", "\u2606", "\u266a", "\u2020", "\u2021", "\u3041", "\u3042", "\u3043", "\u3044", "\u3045", "\u3046", "\u3047", "\u3048", "\u3049", "\u304a", "\u304b", "\u304c", "\u304d", "\u304e", "\u304f", "\u3050", "\u3051", "\u3052", "\u3053", "\u3054", "\u3055", "\u3056", "\u3057", "\u3058", "\u3059", "\u305a", "\u305b", "\u305c", "\u305d", "\u305e", "\u305f", "\u3060", "\u3061", "\u3062", "\u3063", "\u3064", "\u3065", "\u3066", "\u3067", "\u3068", "\u3069", "\u306a", "\u306b", "\u306c", "\u306d", "\u306e", "\u306f", "\u3070", "\u3071", "\u3072", "\u3073", "\u3074", "\u3075", "\u3076", "\u3077", "\u3078", "\u3079", "\u307a", "\u307b", "\u307c", "\u307d", "\u307e", "\u307f", "\u3080", "\u3081", "\u3082", "\u3083", "\u3084", "\u3085", "\u3086", "\u3087", "\u3088", "\u3089", "\u308a", "\u308b", "\u308c", "\u308d", "\u308e", "\u308f", "\u3090", "\u3091", "\u3092", "\u3093", "\u3094", "\u3095", "\u3096", "\u30a1", "\u30a2", "\u30a3", "\u30a4", "\u30a5", "\u30a6", "\u30a7", "\u30a8", "\u30a9", "\u30aa", "\u30ab", "\u30ac", "\u30ad", "\u30ae", "\u30af", "\u30b0", "\u30b1", "\u30b2", "\u30b3", "\u30b4", "\u30b5", "\u30b6", "\u30b7", "\u30b8", "\u30b9", "\u30ba", "\u30bb", "\u30bc", "\u30bd", "\u30be", "\u30bf", "\u30c0", "\u30c1", "\u30c2", "\u30c3", "\u30c4", "\u30c5", "\u30c6", "\u30c7", "\u30c8", "\u30c9", "\u30ca", "\u30cb", "\u30cc", "\u30cd", "\u30ce", "\u30cf", "\u30d0", "\u30d1", "\u30d2", "\u30d3", "\u30d4", "\u30d5", "\u30d6", "\u30d7", "\u30d8", "\u30d9", "\u30da", "\u30db", "\u30dc", "\u30dd", "\u30de", "\u30df", "\u30e0", "\u30e1", "\u30e2", "\u30e3", "\u30e4", "\u30e5", "\u30e6", "\u30e7", "\u30e8", "\u30e9", "\u30ea", "\u30eb", "\u30ec", "\u30ed", "\u30ee", "\u30ef", "\u30f0", "\u30f1", "\u30f2", "\u30f3", "\u30f4", "\u30f5", "\u30f6", "\u30f7", "\u30f8", "\u30f9", "\u30fa", "\u543e", "\u8f29", "\u732b", "\u79c1", "\u50d5", "\u4ffa", "\u541b", "\u5f7c", "\u5973", "\u540d", "\u524d", "\u7121", "\u9ad8", "\u5d0e", "\u585a", "\u9d0e", "\u51db", "\u84ee", "\u8475", "\u7fd4", "\u54b2", "\u685c", "\u831c", "\u6953", "\u96eb", "\u98af", "\u7d2c", "\u51ea", "\u745b", "\u6faa", "\u674f", "\u7d50", "\u82bd", "\u840c", "\u9065", "\u84bc", "\u96bc", "\u62d3", "\u60a0", "\u99ff", "\u9f8d", "\u8000", "\u7409", "\u6dbc", "\u681e", "\u5948", "\u8389", "\u4e43", "\u83ef", "\u51b4", "\u6e1a", "\u7d62", "\u7dbe", "\u971e", "\u6842", "\u77b3", "\u6714", "\u723d", "\u83eb", "\u7fe0", "\u9cf3", "\u96c5", "\u6681", "\u7460", "\u7483", "\u7dcb", "\u7d2b", "\u7425", "\u73c0", "\u6bba", "\u95c7", "\u5149", "\u5f71", "\u5263", "\u9b54", "\u6cd5", "\u52c7", "\u738b", "\u795e", "\u59eb", "\u7687", "\u5e1d", "\u9a0e", "\u58eb", "\u5996", "\u7cbe", "\u4e00", "\u53f3", "\u96e8", "\u5186", "\u97f3", "\u4e0b", "\u706b", "\u82b1", "\u8c9d", "\u5b66", "\u6c17", "\u4e5d", "\u4f11", "\u7389", "\u91d1", "\u7a7a", "\u6708", "\u72ac", "\u898b", "\u53e3", "\u6821", "\u5de6", "\u4e09", "\u5c71", "\u5b50", "\u56db", "\u7cf8", "\u5b57", "\u8033", "\u4e03", "\u8eca", "\u624b", "\u5341", "\u51fa", "\u5c0f", "\u4e0a", "\u68ee", "\u4eba", "\u6c34", "\u6b63", "\u751f", "\u9752", "\u5915", "\u77f3", "\u8d64", "\u5343", "\u5ddd", "\u5148", "\u65e9", "\u8db3", "\u6751", "\u5927", "\u7537", "\u7af9", "\u4e2d", "\u866b", "\u753a", "\u5929", "\u7530", "\u571f", "\u4e8c", "\u65e5", "\u5165", "\u5e74", "\u767d", "\u516b", "\u767e", "\u6587", "\u6728", "\u672c", "\u76ee", "\u7acb", "\u529b", "\u6797", "\u516d", "\u5f15", "\u7fbd", "\u96f2", "\u5712", "\u9060", "\u9ec4", "\u4f55", "\u79d1", "\u590f", "\u5bb6", "\u6b4c", "\u753b", "\u56de", "\u4f1a", "\u6d77", "\u7d75", "\u5916", "\u89d2", "\u697d", "\u6d3b", "\u9593", "\u4e38", "\u5ca9", "\u9854", "\u6c7d", "\u8a18", "\u5e30", "\u5f13", "\u725b", "\u9b5a", "\u4eac", "\u5f37", "\u6559", "\u8fd1", "\u5144", "\u5f62", "\u8a08", "\u5143", "\u8a00", "\u539f", "\u6238", "\u53e4", "\u5348", "\u5f8c", "\u8a9e", "\u5de5", "\u516c", "\u5e83", "\u4ea4", "\u8003", "\u884c", "\u5408", "\u8c37", "\u56fd", "\u9ed2", "\u4eca", "\u624d", "\u7d30", "\u4f5c", "\u7b97", "\u6b62", "\u5e02", "\u77e2", "\u59c9", "\u601d", "\u7d19", "\u5bfa", "\u81ea", "\u6642", "\u5ba4", "\u793e", "\u5f31", "\u9996", "\u79cb", "\u9031", "\u6625", "\u66f8", "\u5c11", "\u5834", "\u8272", "\u98df", "\u5fc3", "\u65b0", "\u89aa", "\u56f3", "\u6570", "\u897f", "\u58f0", "\u661f", "\u6674", "\u5207", "\u96ea", "\u8239", "\u7dda", "\u7d44", "\u8d70", "\u591a", "\u592a", "\u4f53", "\u53f0", "\u5730", "\u6c60", "\u77e5", "\u8336", "\u663c", "\u9577", "\u9ce5", "\u671d", "\u76f4", "\u901a", "\u5f1f", "\u5e97", "\u70b9", "\u96fb", "\u5200", "\u51ac", "\u5f53", "\u6771", "\u7b54", "\u982d", "\u540c", "\u9053", "\u8aad", "\u5185", "\u5357", "\u8089", "\u99ac", "\u8cb7", "\u58f2", "\u9ea6", "\u534a", "\u756a", "\u7236", "\u98a8", "\u5206", "\u7ae0", "\u7528", "\u66dc", "\u6765", "\u7406", "\u8a71", "\u60aa", "\u5b89", "\u6697", "\u533b", "\u59d4", "\u610f", "\u80b2", "\u54e1", "\u9662", "\u98f2", "\u904b", "\u6cf3", "\u99c5", "\u592e", "\u6a2a", "\u5c4b", "\u6e29", "\u5316", "\u8377", "\u754c", "\u958b", "\u968e", "\u5bd2", "\u611f", "\u6f22", "\u9928", "\u5cb8", "\u8d77", "\u671f", "\u5ba2", "\u7a76", "\u6025", "\u7d1a", "\u5bae", "\u7403", "\u53bb", "\u6a4b", "\u696d", "\u66f2", "\u5c40", "\u9280", "\u533a", "\u82e6", "\u5177", "\u4fc2", "\u8efd", "\u8840", "\u6c7a", "\u7814", "\u770c", "\u5eab", "\u6e56", "\u5411", "\u5e78", "\u6e2f", "\u53f7", "\u6839", "\u796d", "\u76bf", "\u4ed5", "\u6b7b", "\u4f7f", "\u59cb", "\u6307", "\u6b6f", "\u8a69", "\u6b21", "\u4e8b", "\u6301", "\u5f0f", "\u5b9f", "\u5199", "\u8005", "\u4e3b", "\u53d6", "\u5b88", "\u9152", "\u53d7", "\u5dde", "\u62fe", "\u7d42", "\u96c6", "\u91cd", "\u5bbf", "\u6240", "\u6691", "\u52a9", "\u662d", "\u6d88", "\u5546", "\u52dd", "\u4e57", "\u690d", "\u7533", "\u8eab", "\u771f", "\u6df1", "\u9032", "\u4e16", "\u6574", "\u6614", "\u5168", "\u76f8", "\u9001", "\u60f3", "\u606f", "\u65cf", "\u4ed6", "\u6253", "\u5bfe", "\u5f85", "\u4ee3", "\u7b2c", "\u984c", "\u70ad", "\u77ed", "\u8ac7", "\u7740", "\u6ce8", "\u67f1", "\u4e01", "\u5e33", "\u8abf", "\u8ffd", "\u5b9a", "\u5ead", "\u7b1b", "\u9244", "\u8ee2", "\u90fd", "\u5ea6", "\u6295", "\u8c46", "\u5cf6", "\u6e6f", "\u767b", "\u7b49", "\u52d5", "\u7ae5", "\u8fb2", "\u6ce2", "\u914d", "\u500d", "\u7bb1", "\u7551", "\u767a", "\u53cd", "\u5742", "\u677f", "\u76ae", "\u5983", "\u5339", "\u6a0b", "\u8868", "\u79d2", "\u75c5", "\u54c1", "\u8ca0", "\u90e8", "\u670d", "\u798f", "\u7269", "\u5e73", "\u8fd4", "\u52c9", "\u5f01", "\u4fdd", "\u679a", "\u5f79", "\u85ac", "\u7531", "\u6cb9", "\u6709", "\u904a", "\u4e88", "\u7f8a", "\u6d0b", "\u8449", "\u967d", "\u69d8", "\u843d", "\u6d41", "\u65c5", "\u4e21", "\u7dd1", "\u793c", "\u548c", "\u611b", "\u6848", "\u4ee5", "\u8863", "\u4f4d", "\u56f2", "\u80c3", "\u5370", "\u82f1", "\u6804", "\u5869", "\u5104", "\u52a0", "\u679c", "\u8ca8", "\u8ab2", "\u6539", "\u68b0", "\u5bb3", "\u8857", "\u5404", "\u899a", "\u5b8c", "\u5b98", "\u7ba1", "\u95a2", "\u89b3", "\u9858", "\u5e0c", "\u5b63", "\u7d00", "\u559c", "\u65d7", "\u5668", "\u6a5f", "\u8b70", "\u6c42", "\u6ce3", "\u6551", "\u7d66", "\u6e05", "\u6319", "\u6f01", "\u5171", "\u5354", "\u93e1", "\u7af6", "\u6975", "\u8a13", "\u8ecd", "\u90e1", "\u5f84", "\u578b", "\u666f", "\u82b8", "\u6b20", "\u5efa", "\u5065", "\u9a13", "\u56fa", "\u529f", "\u597d", "\u5eb7", "\u822a", "\u544a", "\u5dee", "\u83dc", "\u6700", "\u6750", "\u6628", "\u672d", "\u5237", "\u5bdf", "\u53c2", "\u7523", "\u6563", "\u6b8b", "\u6c0f", "\u8a66", "\u5150", "\u6cbb", "\u6ecb", "\u8f9e", "\u9e7f", "\u5931", "\u501f", "\u7a2e", "\u5468", "\u795d", "\u9806", "\u521d", "\u677e", "\u7b11", "\u5531", "\u713c", "\u7167", "\u57ce", "\u7e04", "\u6210", "\u9759", "\u8aac", "\u6298", "\u6d45", "\u5358", "\u6226", "\u9078", "\u7136", "\u4e89", "\u5009", "\u5de3", "\u675f", "\u5074", "\u7d9a", "\u5352", "\u5b6b", "\u5e2f", "\u968a", "\u9054", "\u7f6e", "\u5178", "\u4f1d", "\u706f", "\u50cd", "\u7279", "\u5fb3", "\u6bd2", "\u71b1", "\u5ff5", "\u4fe1", "\u4e0d", "\u4ed8", "\u592b", "\u5e9c", "\u526f", "\u7c89", "\u5175", "\u5225", "\u8fba", "\u5909", "\u4fbf", "\u5305", "\u7267", "\u535a", "\u98ef", "\u98db", "\u5fc5", "\u7968", "\u6a19", "\u5727", "\u79fb", "\u56e0", "\u6c38", "\u55b6", "\u885b", "\u6613", "\u76ca", "\u6db2", "\u6f14", "\u5fdc", "\u5f80", "\u53ef", "\u4eee", "\u4fa1", "\u904e", "\u8cc0", "\u5feb", "\u89e3", "\u683c", "\u78ba", "\u984d", "\u520a", "\u5e79", "\u6163", "\u773c", "\u57fa", "\u5bc4", "\u898f", "\u6280", "\u7fa9", "\u9006", "\u4e45", "\u65e7", "\u5c45", "\u5883", "\u5747", "\u7981", "\u53e5", "\u7fa4", "\u7d4c", "\u6f54", "\u4ef6", "\u5238", "\u967a", "\u691c", "\u9650", "\u73fe", "\u6e1b", "\u6545", "\u500b", "\u8b77", "\u52b9", "\u539a", "\u8015", "\u9271", "\u69cb", "\u8208", "\u8b1b", "\u518d", "\u59bb", "\u63a1", "\u969b", "\u5728", "\u8ca1", "\u7f6a", "\u96d1", "\u9178", "\u8cdb", "\u652f", "\u5fd7", "\u679d", "\u5e2b", "\u8cc7", "\u98fc", "\u793a", "\u4f3c", "\u8b58", "\u820e", "\u8b1d", "\u716e", "\u5c04", "\u6368", "\u91c8", "\u6388", "\u4fee", "\u8ff0", "\u8853", "\u6e96", "\u5e8f", "\u9664", "\u627f", "\u62db", "\u8a3c", "\u6761", "\u72b6", "\u60c5", "\u5e38", "\u7e54", "\u8077", "\u5236", "\u6027", "\u653f", "\u52e2", "\u88fd", "\u7a0e", "\u8cac", "\u7e3e", "\u63a5", "\u8a2d", "\u820c", "\u7d76", "\u92ad", "\u7956", "\u7d20", "\u7dcf", "\u9020", "\u50cf", "\u5897", "\u5247", "\u6e2c", "\u5c5e", "\u7387", "\u640d", "\u9000", "\u8cb8", "\u614b", "\u56e3", "\u65ad", "\u7bc9", "\u5f35", "\u63d0", "\u7a0b", "\u9069", "\u6575", "\u7d71", "\u9285", "\u5c0e", "\u72ec", "\u4efb", "\u80fd", "\u7834", "\u72af", "\u5224", "\u7248", "\u6bd4", "\u80a5", "\u975e", "\u5099", "\u8cbb", "\u6279", "\u79d8", "\u5a66", "\u5bcc", "\u5e03", "\u6b66", "\u5fa9", "\u8907", "\u4ecf", "\u7de8", "\u5893", "\u5831", "\u8c4a", "\u9632", "\u8cbf", "\u66b4", "\u8108", "\u52d9", "\u5922", "\u8ff7", "\u7dbf", "\u6a21", "\u8a33", "\u9810", "\u5bb9", "\u7559", "\u9818", "\u7570", "\u907a", "\u57df", "\u5b87", "\u6620", "\u5ef6", "\u6cbf", "\u6211", "\u7070", "\u62e1", "\u9769", "\u95a3", "\u5272", "\u682a", "\u5e72", "\u5dfb", "\u770b", "\u7c21", "\u5371", "\u673a", "\u63ee", "\u8cb4", "\u7591", "\u5438", "\u4f9b", "\u80f8", "\u90f7", "\u52e4", "\u7b4b", "\u7cfb", "\u656c", "\u8b66", "\u5287", "\u7a74", "\u7d79", "\u6a29", "\u61b2", "\u6e90", "\u53b3", "\u5df1", "\u547c", "\u8aa4", "\u540e", "\u5b5d", "\u7d05", "\u964d", "\u92fc", "\u523b", "\u7a40", "\u9aa8", "\u56f0", "\u7802", "\u5ea7", "\u6e08", "\u88c1", "\u7b56", "\u518a", "\u59ff", "\u81f3", "\u8996", "\u8a5e", "\u8a8c", "\u78c1", "\u5c3a", "\u82e5", "\u6a39", "\u53ce", "\u5b97", "\u5c31", "\u8846", "\u5f93", "\u7e26", "\u7e2e", "\u719f", "\u7d14", "\u51e6", "\u7f72", "\u8af8", "\u5c06", "\u50b7", "\u969c", "\u84b8", "\u91dd", "\u4ec1", "\u5782", "\u63a8", "\u5bf8", "\u76db", "\u8056", "\u8aa0", "\u5ba3", "\u5c02", "\u6cc9", "\u6d17", "\u67d3", "\u5584", "\u5275", "\u594f", "\u5c64", "\u64cd", "\u8535", "\u81d3", "\u5b58", "\u5c0a", "\u5c55", "\u63a2", "\u8a95", "\u6bb5", "\u6696", "\u5024", "\u5b99", "\u5fe0", "\u8457", "\u5e81", "\u9802", "\u8178", "\u6f6e", "\u8cc3", "\u75db", "\u8a0e", "\u515a", "\u7cd6", "\u5f97", "\u5c4a", "\u96e3", "\u4e73", "\u8a8d", "\u7d0d", "\u8133", "\u6d3e", "\u62dd", "\u80cc", "\u80ba", "\u4ff3", "\u73ed", "\u6669", "\u5426", "\u8179", "\u596e", "\u4e26", "\u965b", "\u9589", "\u7247", "\u88dc", "\u66ae", "\u5b9d", "\u8a2a", "\u4ea1", "\u5fd8", "\u68d2", "\u5e55", "\u5bc6", "\u76df", "\u90f5", "\u512a", "\u5e7c", "\u6b32", "\u7fcc", "\u4e71", "\u5375", "\u89a7", "\u88cf", "\u5f8b", "\u81e8", "\u6717", "\u8ad6", "\u4e02", "\u4e04", "\u4e05", "\u4e06", "\u4e07", "\u4e08", "\u4e0c", "\u4e0e", "\u4e0f", "\u4e10", "\u4e11", "\u4e12", "\u4e13", "\u4e14", "\u4e15", "\u4e17", "\u4e18", "\u4e19", "\u4e1a", "\u4e1b", "\u4e1c", "\u4e1d", "\u4e1e", "\u4e1f", "\u4e20", "\u4e22", "\u4e23", "\u4e24", "\u4e25", "\u4e27", "\u4e28", "\u4e29", "\u4e2a", "\u4e2b", "\u4e2c", "\u4e2e", "\u4e2f", "\u4e30", "\u4e31", "\u4e32", "\u4e33", "\u4e34", "\u4e35", "\u4e36", "\u4e37", "\u4e39", "\u4e3a", "\u4e3c", "\u4e3d", "\u4e3e", "\u4e3f", "\u4e40", "\u4e41", "\u4e42", "\u4e44", "\u4e46", "\u4e47", "\u4e48", "\u4e49", "\u4e4a", "\u4e4b", "\u4e4c", "\u4e4d", "\u4e4e", "\u4e4f", "\u4e50", "\u4e51", "\u4e52", "\u4e53", "\u4e54", "\u4e55", "\u4e56", "\u4e58", "\u4e59", "\u4e5a", "\u4e5b", "\u4e5c", "\u4e5e", "\u4e5f", "\u4e60", "\u4e61", "\u4e62", "\u4e63", "\u4e64", "\u4e65", "\u4e66", "\u4e67", "\u4e68", "\u4e69", "\u4e6a", "\u4e6b", "\u4e6c", "\u4e6d", "\u4e6e", "\u4e6f", "\u4e70", "\u4e72", "\u4e74", "\u4e75", "\u4e76", "\u4e77", "\u4e78", "\u4e79", "\u4e7a", "\u4e7b", "\u4e7c", "\u4e7d", "\u4e7e", "\u4e7f", "\u4e80", "\u4e81", "\u4e82", "\u4e83", "\u4e84", "\u4e85", "\u4e86", "\u4e87", "\u4e8a", "\u4e8d", "\u4e8e", "\u4e8f", "\u4e90", "\u4e91", "\u4e92", "\u4e93", "\u4e94", "\u4e95", "\u4e96", "\u4e97", "\u4e98", "\u4e99", "\u4e9a", "\u4e9b", "\u4e9c", "\u4e9d", "\u4e9e", "\u4e9f", "\u4ea0", "\u4ea2", "\u4ea3", "\u4ea5", "\u4ea6", "\u4ea7", "\u4ea8", "\u4ea9", "\u4eaa", "\u4eab", "\u4ead", "\u4eae", "\u4eaf", "\u4eb0", "\u4eb1", "\u4eb2", "\u4eb3", "\u4eb4", "\u4eb5", "\u4eb6", "\u4eb7", "\u4eb8", "\u4eb9", "\u4ebb", "\u4ebc", "\u4ebd", "\u4ebe", "\u4ebf", "\u4ec0", "\u4ec2", "\u4ec3", "\u4ec4", "\u4ec5", "\u4ec6", "\u4ec7", "\u4ec8", "\u4ec9", "\u4ecb", "\u4ecc", "\u4ecd", "\u4ece", "\u4ed0", "\u4ed1", "\u4ed2", "\u4ed3", "\u4ed4", "\u4ed7", "\u4ed9", "\u4eda", "\u4edb", "\u4edc", "\u4edd", "\u4ede", "\u4edf", "\u4ee0", "\u4ee1", "\u4ee2", "\u4ee4", "\u4ee6", "\u4ee7", "\u4ee8", "\u4ee9", "\u4eea", "\u4eeb", "\u4eec", "\u4eed", "\u4eef", "\u4ef0", "\u4ef1", "\u4ef2", "\u4ef3", "\u4ef4", "\u4ef5", "\u4ef7", "\u4ef8", "\u4ef9", "\u4efa", "\u4efc", "\u4efd", "\u4efe", "\u4eff", "\u4f00", "\u4f01", "\u4f02", "\u4f03", "\u4f04", "\u4f05", "\u4f06", "\u4f07", "\u4f08", "\u4f09", "\u4f0a", "\u4f0b", "\u4f0c", "\u4f0d", "\u4f0e", "\u4f0f", "\u4f10", "\u4f12", "\u4f13", "\u4f14", "\u4f15", "\u4f16", "\u4f17", "\u4f18", "\u4f19", "\u4f1b", "\u4f1c", "\u4f1e", "\u4f1f", "\u4f20", "\u4f21", "\u4f22", "\u4f23", "\u4f24", "\u4f25", "\u4f26", "\u4f27", "\u4f28", "\u4f29", "\u4f2a", "\u4f2b", "\u4f2c", "\u4f2d", "\u4f2e", "\u4f2f", "\u4f30", "\u4f31", "\u4f32", "\u4f33", "\u4f34", "\u4f35", "\u4f36", "\u4f37", "\u4f38", "\u4f39", "\u4f3a", "\u4f3b", "\u4f3d", "\u4f3e", "\u4f3f", "\u4f40", "\u4f41", "\u4f42", "\u4f43", "\u4f44", "\u4f45", "\u4f46", "\u4f47", "\u4f48", "\u4f49", "\u4f4a", "\u4f4b", "\u4f4c", "\u4f4e", "\u4f4f", "\u4f50", "\u4f51", "\u4f52", "\u4f54", "\u4f56", "\u4f57", "\u4f58", "\u4f59", "\u4f5a", "\u4f5b", "\u4f5d", "\u4f5e", "\u4f5f", "\u4f60", "\u4f61", "\u4f62", "\u4f63", "\u4f64", "\u4f65", "\u4f66", "\u4f67", "\u4f68", "\u4f69", "\u4f6a", "\u4f6b", "\u4f6c", "\u4f6d", "\u4f6e", "\u4f6f", "\u4f70", "\u4f71", "\u4f72", "\u4f73", "\u4f74", "\u4f75", "\u4f76", "\u4f77", "\u4f78", "\u4f79", "\u4f7a", "\u4f7b", "\u4f7c", "\u4f7d", "\u4f7e", "\u4f80", "\u4f81", "\u4f82", "\u4f83", "\u4f84", "\u4f85", "\u4f86", "\u4f87", "\u4f88", "\u4f89", "\u4f8a", "\u4f8b", "\u4f8c", "\u4f8d", "\u4f8e", "\u4f8f", "\u4f90", "\u4f91", "\u4f92", "\u4f93", "\u4f94", "\u4f95", "\u4f96", "\u4f97", "\u4f98", "\u4f99", "\u4f9a", "\u4f9c", "\u4f9d", "\u4f9e", "\u4f9f", "\u4fa0", "\u4fa2", "\u4fa3", "\u4fa4", "\u4fa5", "\u4fa6", "\u4fa7", "\u4fa8", "\u4fa9", "\u4faa", "\u4fab", "\u4fac", "\u4fad", "\u4fae", "\u4faf", "\u4fb0", "\u4fb1", "\u4fb2", "\u4fb3", "\u4fb4", "\u4fb5", "\u4fb6", "\u4fb7", "\u4fb8", "\u4fb9", "\u4fba", "\u4fbb", "\u4fbc", "\u4fbd", "\u4fbe", "\u4fc0", "\u4fc1", "\u4fc3", "\u4fc4", "\u4fc5", "\u4fc6", "\u4fc7", "\u4fc8", "\u4fc9", "\u4fca", "\u4fcb", "\u4fcc", "\u4fcd", "\u4fce", "\u4fcf", "\u4fd0", "\u4fd1", "\u4fd2", "\u4fd3", "\u4fd4", "\u4fd5", "\u4fd6", "\u4fd7", "\u4fd8", "\u4fd9", "\u4fda", "\u4fdb", "\u4fdc", "\u4fde", "\u4fdf", "\u4fe0", "\u4fe2", "\u4fe3", "\u4fe4", "\u4fe5", "\u4fe6", "\u4fe7", "\u4fe8", "\u4fe9", "\u4fea", "\u4feb", "\u4fec", "\u4fed", "\u4fef", "\u4ff0", "\u4ff1", "\u4ff2", "\u4ff4", "\u4ff5", "\u4ff6", "\u4ff7", "\u4ff8", "\u4ff9", "\u4ffb", "\u4ffc", "\u4ffd", "\u4ffe", "\u4fff", "\u5000", "\u5001", "\u5002", "\u5003", "\u5004", "\u5005", "\u5006", "\u5007", "\u5008", "\u500a", "\u500c", "\u500e", "\u500f", "\u5010", "\u5011", "\u5012", "\u5013", "\u5014", "\u5015", "\u5016", "\u5017", "\u5018", "\u5019", "\u501a", "\u501b", "\u501c", "\u501d", "\u501e", "\u5020", "\u5021", "\u5022", "\u5023", "\u5025", "\u5026", "\u5027", "\u5028", "\u5029", "\u502a", "\u502b", "\u502c", "\u502d", "\u502e", "\u502f", "\u5030", "\u5031", "\u5032", "\u5033", "\u5034", "\u5035", "\u5036", "\u5037", "\u5038", "\u5039", "\u503a", "\u503b", "\u503c", "\u503d", "\u503e", "\u503f", "\u5040", "\u5041", "\u5042", "\u5043", "\u5044", "\u5045", "\u5046", "\u5047", "\u5048", "\u5049", "\u504a", "\u504b", "\u504c", "\u504d", "\u504e", "\u504f", "\u5050", "\u5051", "\u5052", "\u5053", "\u5054", "\u5055", "\u5056", "\u5057", "\u5058", "\u5059", "\u505a", "\u505b", "\u505c", "\u505d", "\u505e", "\u505f", "\u5060", "\u5061", "\u5062", "\u5063", "\u5064", "\u5066", "\u5067", "\u5068", "\u5069", "\u506a", "\u506b", "\u506c", "\u506d", "\u506e", "\u506f", "\u5070", "\u5071", "\u5072", "\u5073", "\u5075", "\u5076", "\u5077", "\u5078", "\u5079", "\u507a", "\u507b", "\u507c", "\u507d", "\u507e", "\u507f", "\u5080", "\u5081", "\u5082", "\u5083", "\u5084", "\u5085", "\u5086", "\u5087", "\u5088", "\u5089", "\u508a", "\u508b", "\u508c", "\u508d", "\u508e", "\u508f", "\u5090", "\u5091", "\u5092", "\u5093", "\u5094", "\u5095", "\u5096", "\u5097", "\u5098", "\u509a", "\u509b", "\u509c", "\u509d", "\u509e", "\u509f", "\u50a0", "\u50a1", "\u50a2", "\u50a3", "\u50a4", "\u50a5", "\u50a6", "\u50a7", "\u50a8", "\u50a9", "\u50aa", "\u50ab", "\u50ac", "\u50ad", "\u50ae", "\u50af", "\u50b0", "\u50b1", "\u50b2", "\u50b3", "\u50b4", "\u50b5", "\u50b6", "\u50b8", "\u50b9", "\u50ba", "\u50bb", "\u50bc", "\u50bd", "\u50be", "\u50bf", "\u50c0", "\u50c1", "\u50c2", "\u50c3", "\u50c4", "\u50c5", "\u50c6", "\u50c7", "\u50c8", "\u50c9", "\u50ca", "\u50cb", "\u50cc", "\u50ce", "\u50d0", "\u50d1", "\u50d2", "\u50d3", "\u50d4", "\u50d6", "\u50d7", "\u50d8", "\u50d9", "\u50da", "\u50db", "\u50dc", "\u50dd", "\u50de", "\u50df", "\u50e0", "\u50e1", "\u50e2", "\u50e3", "\u50e4", "\u50e5", "\u50e6", "\u50e7", "\u50e8", "\u50e9", "\u50ea", "\u50eb", "\u50ec", "\u50ed", "\u50ee", "\u50ef", "\u50f0", "\u50f1", "\u50f2", "\u50f3", "\u50f4", "\u50f5", "\u50f6", "\u50f7", "\u50f8", "\u50f9", "\u50fa", "\u50fb", "\u50fc", "\u50fd", "\u50fe", "\u50ff", "\u5100", "\u5101", "\u5102", "\u5103", "\u5105", "\u5106", "\u5107", "\u5108", "\u5109", "\u510a", "\u510b", "\u510c", "\u510d"]
VOCAB_MAP = {ch: idx for idx, ch in enumerate(VOCAB_LIST)}
PAD_ID = VOCAB_MAP.get("[PAD]", 0)
UNK_ID = VOCAB_MAP.get("[UNK]", 1)
BOS_ID = VOCAB_MAP.get("[BOS]", 2)
EOS_ID = VOCAB_MAP.get("[EOS]", 3)
#
def tokenize_char(text, max_len=256):
    ids = []
    for ch in text[:max_len]:
        ids.append(VOCAB_MAP.get(ch, UNK_ID))
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return ids, length
#
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=96, hidden_dim=384):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, hidden_dim, bias=False)
    def forward(self, input_ids):
        return self.projection(self.word_embeddings(input_ids))
#
class NarrativeNanoEncoder(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=384, num_layers=8, num_heads=6, intermediate_dim=1536, dropout=0.1):
        super().__init__()
        self.embedding = FactorizedEmbedding(vocab_size, 96, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)
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
        self.head_case = nn.Linear(hidden_dim, 10)
        self.head_epistemic = nn.Linear(hidden_dim, 1)
        self.head_event_action = nn.Linear(hidden_dim, 6) # None, Acquire, Drop, Move, Speak, StateChange
        self.head_entity = nn.Linear(hidden_dim, 4) # O, B-ENT, I-ENT, E-ENT
        self.head_connective = nn.Linear(hidden_dim, 5) # None, Causal, Adversative, Temporal, Additive
#
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(x)
        out_modality = self.head_modality(h)
        out_offset = self.head_offset(h)
        out_label = self.head_label(h)
        out_case = self.head_case(h)
        out_epistemic = torch.sigmoid(self.head_epistemic(h))
        out_event_action = self.head_event_action(h)
        out_entity = self.head_entity(h)
        out_connective = self.head_connective(h)
        return out_modality, out_offset, out_label, out_case, out_epistemic, out_event_action, out_entity, out_connective
#
print("=== 2. Loading Literary & Synthetic Datasets ===")
AOZORA_WORKS = [
    ("\u3053\u3053\u308d", "\u590f\u76ee\u6f31\u77f3", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/773_ruby_5968/773_ruby_5968.txt"),
    ("\u574a\u3063\u3061\u3083\u3093", "\u590f\u76ee\u6f31\u77f3", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/752_ruby_2438/752_ruby_2438.txt"),
    ("\u7f85\u751f\u9580", "\u82a5\u5ddd\u9f8d\u4e4b\u4ecb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/127_ruby_150/127_ruby_150.txt"),
    ("\u8d70\u308c\u30e1\u30ed\u30b9", "\u592a\u5bb0\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1567_ruby_4948/1567_ruby_4948.txt"),
    ("\u9280\u6cb3\u9244\u9053\u306e\u591c", "\u5bae\u6ca2\u8ce2\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000081/files/43737_ruby_17918/43737_ruby_17918.txt"),
    ("\u6ab8\u6aac", "\u68b6\u4e95\u57fa\u6b21\u90ce", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000074/files/427_ruby_143/427_ruby_143.txt"),
    ("\u4eba\u9593\u5931\u683c", "\u592a\u5bb0\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/301_ruby_5915/301_ruby_5915.txt"),
    ("\u6ce8\u6587\u306e\u591a\u3044\u6599\u7406\u5e97", "\u5bae\u6ca2\u8ce2\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000081/files/1920_ruby_18525/1920_ruby_18525.txt"),
    ("\u5c71\u6708\u8a18", "\u4e2d\u5cf6\u6566", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000119/files/624_ruby_1444/624_ruby_1444.txt"),
    ("\u8718\u86db\u306e\u7cf8", "\u82a5\u5ddd\u9f8d\u4e4b\u4ecb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/92_ruby_164/92_ruby_164.txt"),
    ("\u659c\u967d", "\u592a\u5bb0\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1565_ruby_8220/1565_ruby_8220.txt"),
    ("\u821e\u59eb", "\u68ee\u9dd7\u5916", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000129/files/2059_ruby_19889/2059_ruby_19889.txt"),
    ("\u91ce\u83ca\u306e\u5893", "\u4f0a\u85e4\u5de6\u5343\u592b", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000059/files/1063_ruby_4210/1063_ruby_4210.txt"),
    ("\u6932\u306e\u6728\u306e\u9670", "\u6a03\u6a39\u4e00\u751f", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000216/files/1070_ruby_4213/1070_ruby_4213.txt"),
    ("\u6c41\u7269", "\u9b6f\u5c71\u4eba", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/001403/files/49986_ruby_37779/49986_ruby_37779.txt"),
    ("\u9ad8\u702c\u821f", "\u68ee\u9dd7\u5916", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000129/files/691_ruby_1595/691_ruby_1595.txt"),
    # RUN-018: 8 New Verified Masterpieces (Natsume Soseki, Dazai Osamu, Akutagawa Ryunosuke, Miyazawa Kenji)
    ("\u502b\u6566\u5854", "\u590f\u76ee\u6f31\u77f3", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/1076_ruby_4527/1076_ruby_4527.txt"),
    ("\u4e09\u56db\u90ce", "\u590f\u76ee\u6f31\u77f3", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/794_ruby_4237/794_ruby_4237.txt"),
    ("\u7af9\u9752", "\u592a\u5bb0\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1047_ruby_20129/1047_ruby_20129.txt"),
    ("\u8d70\u3089\u306e\u540d\u99ac", "\u592a\u5bb0\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000035/files/1059_ruby_4748/1059_ruby_4748.txt"),
    ("\u6843\u592a\u90ce", "\u82a5\u5ddd\u9f8d\u4e4b\u4ecb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/100_ruby_1154/100_ruby_1154.txt"),
    ("\u6bdb\u5229\u5148\u751f", "\u82a5\u5ddd\u9f8d\u4e4b\u4ecb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000879/files/101_ruby_857/101_ruby_857.txt"),
    ("\u6625\u3068\u4fee\u7f85", "\u5bae\u6ca2\u8ce2\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000081/files/1058_ruby_4709/1058_ruby_4709.txt"),
    ("\u4e8c\u4eba\u306e\u5f79\u4eba", "\u5bae\u6ca2\u8ce2\u6cbb", "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000081/files/1064_ruby_19929/1064_ruby_19929.txt"),
]
#
CASE_PARTICLES = [
    ("\u304c", 0), ("\u3092", 1), ("\u306b", 2), ("\u3067", 3), ("\u3068", 4),
    ("\u3088\u308a", 5), ("\u304b\u3089", 6), ("\u307e\u3067", 7), ("\u3078", 8), ("\u306e", 9)
]
#
def clean_aozora_text(raw_text):
    lines = raw_text.splitlines()
    body_lines = []
    dash_count = 0
    in_header = True
    for line in lines:
        s = line.strip()
        if s.startswith("-------"):
            dash_count += 1
            if dash_count == 2:
                in_header = False
            continue
        if in_header and dash_count < 2:
            continue
        if s.startswith("\u5e95\u672c\uff1a") or s.startswith("\uff3b\uff03\u672c\u6587\u7d42\u308f\u308a\uff3d"):
            break
        body_lines.append(line)
    text = "\n".join(body_lines)
    text = re.sub(r'\uff3b\uff03[^\uff3d]*\uff3d', '', text)
    text = re.sub(r'\uff5c([^\u300a\n]+)\u300a([^\u300b\n]+)\u300b', r'\1', text)
    text = re.sub(r'([\u4e00-\u9fa0\u3005\u3006\u30f5\u30f6]+)\u300a([^\u300b\n]+)\u300b', r'\1', text)
    text = re.sub(r'\u300a[^\u300b\n]*\u300b', '', text)
    text = text.replace('\uff5c', '')
    return text
#
def segment_novel_text(text):
    dialogue_pattern = re.compile(r'(\u300c[^\u300d]*\u300d|\u300e[^\u300f]*\u300f)')
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = dialogue_pattern.split(line)
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if (tok.startswith('\u300c') and tok.endswith('\u300d')) or (tok.startswith('\u300e') and tok.endswith('\u300f')):
                results.append((tok, True))
            else:
                sents = re.findall(r'([^\u3002\uff01\uff1f!]*[\u3002\uff01\uff1f!])', tok)
                rem = re.sub(r'([^\u3002\uff01\uff1f!]*[\u3002\uff01\uff1f!])', '', tok).strip()
                for s in sents:
                    if s.strip():
                        results.append((s.strip(), False))
                if rem:
                    results.append((rem, False))
    return results
#
raw_samples = []
for title, author, url in AOZORA_WORKS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode('shift_jis', errors='ignore')
            cleaned = clean_aozora_text(raw)
            segs = segment_novel_text(cleaned)
            print(f"  [+] Loaded {title} ({author}): {len(segs):,} segments")
            for s, is_dia in segs:
                if 4 <= len(s) <= 120:
                    raw_samples.append((s, is_dia))
    except Exception as e:
        print(f"  [!] Error fetching {title}: {e}")
#
# RUN-018: Generate 58,000 synthetic sentences for Typo Recovery Layer 2 & Logic Guard POV
import random
subjects = ["俺", "私", "僕", "彼女", "彼", "勇者", "主人公", "ギルドマスター", "先輩", "後輩", "エリス", "アリア", "魔王", "少女", "探偵", "騎士", "賢者", "商人", "王女"]
objects = ["ステータス画面", "スキル一覧", "古代の魔導書", "真実の鍵", "スマートフォン", "冷めたコーヒー", "依頼書", "聖剣", "黒い短剣", "壊れた時計", "水晶玉", "手紙", "記憶の欠片", "古びた地図"]
locations = ["ギルドの酒場", "暗いダンジョンの中層", "オフィスの静寂", "放課後の教室", "地下闘技場", "壊れかけた神殿", "薄暗い自室", "静かな図書室", "霧深い森の奥", "城のバルコニー", "広場の中央"]
actions = ["静かに睨みつけた", "ため息をつきながら開いた", "素早くポケットに隠した", "信じられない思いで見つめた", "迷わず手に取った", "呟いて立ち上がった", "恐る恐る触れた", "しっかりと握りしめた"]

# Typo and homophone awareness templates (Layer 2)
typo_aware_dialogues = [
    ("「少々お待ちくだしあ」と店員は頭を下げた。", True),
    ("「こんちには、今日も良い天気ですね」と挨拶した。", True),
    ("「その件について、ｔお確認いたします」と答えた。", True),
    ("記者たちが急いで汽車で帰社していった。", False),
    ("彼女の意図を汲み取って、赤い糸を手繰り寄せた。", False),
]

for _ in range(58000):
    loc = random.choice(locations)
    sub = random.choice(subjects)
    obj = random.choice(objects)
    act = random.choice(actions)
    s = f"{loc}で、{sub}は{obj}を{act}。"
    raw_samples.append((s, False))

for _ in range(200):
    raw_samples.extend(typo_aware_dialogues)
#
print(f"Total Dataset Samples: {len(raw_samples):,}")
#
class LiteraryDataset(Dataset):
    def __init__(self, samples, max_len=256):
        self.samples = samples
        self.max_len = max_len
#
    def __len__(self):
        return len(self.samples)
#
    def __getitem__(self, idx):
        text, is_dia = self.samples[idx]
        tokens, length = tokenize_char(text, self.max_len)
        target_mod = [1 if is_dia else 0] * self.max_len
        for i in range(length, self.max_len):
            target_mod[i] = 0
#
        target_off = [32] * self.max_len
        target_lbl = [2] * self.max_len
        target_case = [-100] * self.max_len
        target_epi = [0.0] * self.max_len
        target_act = [0] * self.max_len # 0: None, 1: Acquire, 2: Drop, 3: Move, 4: Speak, 5: StateChange
        target_ent = [0] * self.max_len # 0: O, 1: B-ENT, 2: I-ENT, 3: E-ENT
#
        root_idx = max(0, length - 2) if length >= 2 else 0
        target_lbl[root_idx] = 1
#
        # Simple entity heuristic (proper noun candidates: Katakana spans / Kansei names)
        ent_matches = list(re.finditer(r'([A-Z][a-z]+|[\u30a1-\u30f6]{2,}|先生|メロス|セリヌンティウス|カンパネルラ|ジョバンニ)', text))
        for em in ent_matches:
            s_idx, e_idx = em.start(), em.end()
            if s_idx < length:
                target_ent[s_idx] = 1 # B-ENT
                for ei in range(s_idx + 1, min(length, e_idx)):
                    target_ent[ei] = 2 # I-ENT
#
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
#
        # Action classification heuristic for supervision
        if any(w in text for w in ["拾っ", "手に入れ", "奪っ", "取っ", "受け取"]):
            target_act[root_idx] = 1
        elif any(w in text for w in ["落とし", "失っ", "手放し", "捨て"]):
            target_act[root_idx] = 2
        elif any(w in text for w in ["向かっ", "歩い", "走っ", "訪れ", "旅立"]):
            target_act[root_idx] = 3
        elif is_dia:
            target_act[root_idx] = 4
        elif any(w in text for w in ["壊れ", "変化", "倒れ", "目覚め"]):
            target_act[root_idx] = 5
#
        epi_val = 0.1 if is_dia else (0.8 if any(k in text for k in ["思っ", "感じ", "悲し", "悔し", "怒り"]) else 0.2)
        for i in range(length):
            target_epi[i] = epi_val
#
        target_conn = [0] * self.max_len # 0: None, 1: Causal, 2: Adversative, 3: Temporal, 4: Additive
        if any(text.startswith(w) for w in ["だから", "それゆえ", "したがって", "そのため", "ゆえに"]):
            target_conn[0] = 1
        elif any(text.startswith(w) for w in ["しかし", "だが", "けれども", "ところが", "とはいえ"]):
            target_conn[0] = 2
        elif any(text.startswith(w) for w in ["その時", "翌朝", "数日後", "やがて", "まもなく", "しばらくして"]):
            target_conn[0] = 3
        elif any(text.startswith(w) for w in ["また", "さらに", "その上", "加えて"]):
            target_conn[0] = 4
#
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(target_mod, dtype=torch.long),
            torch.tensor(target_off, dtype=torch.long),
            torch.tensor(target_lbl, dtype=torch.long),
            torch.tensor(target_case, dtype=torch.long),
            torch.tensor(target_epi, dtype=torch.float32),
            torch.tensor(target_act, dtype=torch.long),
            torch.tensor(target_ent, dtype=torch.long),
            torch.tensor(target_conn, dtype=torch.long)
        )
#
val_size = min(1000, int(len(raw_samples) * 0.1))
train_samples = raw_samples[:-val_size]
val_samples = raw_samples[-val_size:]
#
train_dataset = LiteraryDataset(train_samples)
val_dataset = LiteraryDataset(val_samples)
#
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#
print("=== 3. Model Architecture & Initialization ===")
model = NarrativeNanoEncoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")
#
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
num_epochs = 4
total_steps = num_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
#
criterion_ce = nn.CrossEntropyLoss()
criterion_case = nn.CrossEntropyLoss(ignore_index=-100)
criterion_mse = nn.MSELoss()
#
print(f"=== 4. Executing Training Pipeline ({num_epochs} Epochs) ===")
start_time = time.time()
#
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    mod_correct = 0
    mod_total = 0
    case_correct = 0
    case_total = 0
#
    for batch in train_loader:
        b_input, b_mod, b_off, b_lbl, b_case, b_epi, b_act, b_ent, b_conn = [t.to(device) for t in batch]
        optimizer.zero_grad()
#
        out_mod, out_off, out_lbl, out_case, out_epi, out_act, out_ent, out_conn = model(b_input)
#
        loss_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
        loss_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
        loss_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
        loss_case = criterion_case(out_case.view(-1, 10), b_case.view(-1))
        loss_epi = criterion_mse(out_epi.squeeze(-1), b_epi)
        loss_act = criterion_ce(out_act.view(-1, 6), b_act.view(-1))
        loss_ent = criterion_ce(out_ent.view(-1, 4), b_ent.view(-1))
        loss_conn = criterion_ce(out_conn.view(-1, 5), b_conn.view(-1))
#
        loss = 1.0 * loss_mod + 0.5 * loss_off + 0.5 * loss_lbl + 1.2 * loss_case + 0.5 * loss_epi + 1.0 * loss_act + 1.0 * loss_ent + 0.8 * loss_conn
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
#
        total_loss += loss.item()
#
        pred_mod = out_mod.argmax(dim=-1)
        mod_correct += (pred_mod == b_mod).sum().item()
        mod_total += b_mod.numel()
#
        mask = (b_case != -100)
        if mask.any():
            pred_case = out_case.argmax(dim=-1)
            case_correct += (pred_case[mask] == b_case[mask]).sum().item()
            case_total += mask.sum().item()
#
    avg_loss = total_loss / len(train_loader)
    mod_acc = (mod_correct / mod_total) * 100.0
    case_acc = (case_correct / max(1, case_total)) * 100.0
#
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            b_input, b_mod, b_off, b_lbl, b_case, b_epi, b_act, b_ent, b_conn = [t.to(device) for t in batch]
            out_mod, out_off, out_lbl, out_case, out_epi, out_act, out_ent, out_conn = model(b_input)
            loss_mod = criterion_ce(out_mod.view(-1, 2), b_mod.view(-1))
            loss_off = criterion_ce(out_off.view(-1, 65), b_off.view(-1))
            loss_lbl = criterion_ce(out_lbl.view(-1, 8), b_lbl.view(-1))
            loss_case = criterion_case(out_case.view(-1, 10), b_case.view(-1))
            loss_epi = criterion_mse(out_epi.squeeze(-1), b_epi)
            loss_act = criterion_ce(out_act.view(-1, 6), b_act.view(-1))
            loss_ent = criterion_ce(out_ent.view(-1, 4), b_ent.view(-1))
            loss_conn = criterion_ce(out_conn.view(-1, 5), b_conn.view(-1))
            v_loss = 1.0 * loss_mod + 0.5 * loss_off + 0.5 * loss_lbl + 1.2 * loss_case + 0.5 * loss_epi + 1.0 * loss_act + 1.0 * loss_ent + 0.8 * loss_conn
            val_loss += v_loss.item()
    avg_val_loss = val_loss / len(val_loader)
#
    print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Modality Acc: {mod_acc:.2f}% | Case Acc: {case_acc:.2f}%")
#
elapsed = time.time() - start_time
print(f"=== Training Completed in {elapsed:.2f} seconds ===")
#
# Sample predictions
model.eval()
test_sentences = [
    ("\u300c\u79c1\u306f\u305d\u306e\u4eba\u3092\u5e38\u306b\u5148\u751f\u3068\u547c\u3093\u3067\u3044\u305f\u3002\u300d", True),
    ("\u4e0b\u4eba\u306f\u7f85\u751f\u9580\u306e\u4e0b\u3067\u96e8\u3084\u307f\u3092\u5f85\u3063\u3066\u3044\u305f\u3002", False),
    ("\u305d\u306e\u6642\u3001\u30e1\u30ed\u30b9\u306f\u6fc0\u6012\u3057\u305f\u3002\u5fc5\u305a\u90aa\u667a\u66b4\u8650\u306e\u738b\u3092\u9664\u304b\u306a\u3051\u308c\u3070\u306a\u3089\u306c\u3068\u6c7a\u610f\u3057\u305f\u3002", False),
]
print("=== 5. Model Predictions ===")
with torch.no_grad():
    for sent, expected_dia in test_sentences:
        t_ids, t_len = tokenize_char(sent, 256)
        inp = torch.tensor([t_ids], dtype=torch.long, device=device)
        o_mod, o_off, o_lbl, o_case, o_epi, o_act, o_ent, o_conn = model(inp)
        dia_prob = torch.softmax(o_mod[0, :t_len], dim=-1)[:, 1].mean().item()
        epi_avg = o_epi[0, :t_len].mean().item()
        act_pred = o_act[0, :t_len].argmax(dim=-1).tolist()
        ent_pred = o_ent[0, :t_len].argmax(dim=-1).tolist()
        conn_pred = o_conn[0, 0].argmax(dim=-1).item()
        print(f"Text: {sent}")
        print(f"  Dialogue Prob: {dia_prob:.4f} | Epistemic POV: {epi_avg:.4f} | Connective: {conn_pred} | Actions: {set(act_pred)} | Entities: {set(ent_pred)}")
#
# ONNX Export & Dynamic INT8 Quantization
print("=== 6. ONNX Model Export & Quantization ===")
model_cpu = model.cpu()
dummy_input = torch.randint(0, 2048, (1, 256), dtype=torch.long)
fp32_onnx_path = "/tmp/narrative_nano_pro_v14_fp32.onnx"
int8_onnx_path = "/tmp/narrative_nano_pro_v14_qat_int8.onnx"

torch.onnx.export(
    model_cpu,
    dummy_input,
    fp32_onnx_path,
    input_names=["input_ids"],
    output_names=["modality", "offset", "label", "case", "epistemic", "event_action", "entity", "connective"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
    opset_version=17,
    dynamo=False
)

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_model = onnx.load(fp32_onnx_path)
onnx.checker.check_model(onnx_model)
fp32_size = os.path.getsize(fp32_onnx_path)
print(f"[SUCCESS] FP32 ONNX Exported: {fp32_onnx_path} ({fp32_size / (1024*1024):.2f} MB)")

print("[*] Quantizing to INT8...")
quantize_dynamic(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    extra_options={"DisableShapeInference": True}
)
int8_size = os.path.getsize(int8_onnx_path)
comp_ratio = round((1 - int8_size / fp32_size) * 100, 1)
print(f"[SUCCESS] INT8 ONNX Exported: {int8_onnx_path} ({int8_size / (1024*1024):.2f} MB, {comp_ratio}% compression)")

# Google Drive Persistence Protocol (Evaporation Prevention)
drive_dir = "/content/drive/MyDrive/worldcraft_models"
if os.path.exists("/content/drive/MyDrive"):
    os.makedirs(f"{drive_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{drive_dir}/onnx", exist_ok=True)
    os.makedirs(f"{drive_dir}/logs", exist_ok=True)
    import shutil
    shutil.copy2(int8_onnx_path, f"{drive_dir}/onnx/narrative_nano_pro_v14_qat_int8.onnx")
    torch.save(model_cpu.state_dict(), f"{drive_dir}/checkpoints/scaled_narrative_nano_v14_pro.pt")
    print(f"[✓] Persisted model and weights to Google Drive: {drive_dir}")
else:
    print("[i] Google Drive not mounted in non-interactive environment; saving local checkpoint in /tmp/worldcraft_models")
    os.makedirs("/tmp/worldcraft_models/checkpoints", exist_ok=True)
    torch.save(model_cpu.state_dict(), "/tmp/worldcraft_models/checkpoints/scaled_narrative_nano_v14_pro.pt")

# Upload artifact for automated download helper
up_res = subprocess.getoutput(f"curl -s -F 'file=@{int8_onnx_path}' https://tmpfiles.org/api/v1/upload")
print(f"AUTO_DOWNLOAD_URL:{up_res}")
print("=== ONNX READY FOR ARTIFACT COLLECTION ===")