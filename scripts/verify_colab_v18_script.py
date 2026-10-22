import os
import sys
import json
import gzip
import base64

def verify():
    script_path = "scripts/build_colab_v18_script.py"
    assert os.path.exists(script_path), f"File not found: {script_path}"
    
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    corpus_found = False
    bench_found = False
    vocab_found = False

    for line in code.split("\n"):
        if line.startswith("CORPUS_B64 = "):
            b64_str = line.split(" = ")[1].strip("\"'")
            raw = gzip.decompress(base64.b64decode(b64_str)).decode("utf-8")
            samples = [json.loads(l) for l in raw.strip().split("\n") if l.strip()]
            print(f"[OK] Embedded Corpus decoded: {len(samples):,} samples")
            assert len(samples) == 9220, f"Expected 9220 samples, got {len(samples)}"
            corpus_found = True
        elif line.startswith("BENCHMARK_B64 = "):
            b64_str = line.split(" = ")[1].strip("\"'")
            raw = gzip.decompress(base64.b64decode(b64_str)).decode("utf-8")
            samples = [json.loads(l) for l in raw.strip().split("\n") if l.strip()]
            print(f"[OK] Embedded Benchmark decoded: {len(samples):,} samples")
            assert len(samples) == 100, f"Expected 100 samples, got {len(samples)}"
            bench_found = True
        elif line.startswith("VOCAB_B64 = "):
            b64_str = line.split(" = ")[1].strip("\"'")
            vocab = json.loads(base64.b64decode(b64_str).decode("utf-8"))
            print(f"[OK] Embedded Vocab decoded: {len(vocab):,} tokens")
            assert len(vocab) == 2048, f"Expected 2048 tokens, got {len(vocab)}"
            vocab_found = True

    assert corpus_found and bench_found and vocab_found, "All embedded data must be present"
    print("\n[SUCCESS] All embedded components verified perfectly!")

if __name__ == "__main__":
    verify()
