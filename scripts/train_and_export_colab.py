#!/usr/bin/env python3
"""
narrative-nano/scripts/train_and_export_colab.py

Orchestrates Google Colab T4 GPU execution for Narrative-Nano training,
knowledge distillation, and ONNX model export using mcp_server_colab_exec.
"""

import os
import sys
import json
import time
from mcp_server_colab_exec.server import colab_execute_notebook

def main():
    print("[*] Reading colab_narrative_nano_pipeline.py...")
    script_path = os.path.join(os.path.dirname(__file__), "colab_narrative_nano_pipeline.py")
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "models"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Allocating Google Colab T4 GPU Runtime and executing pipeline...")
    print(f"[*] Output directory for ONNX artifacts: {output_dir}")
    start_time = time.time()

    raw_result = colab_execute_notebook(
        code=code,
        output_dir=output_dir,
        accelerator="T4",
        timeout=300
    )

    elapsed = time.time() - start_time
    result = json.loads(raw_result)

    print(f"[+] Colab execution completed in {elapsed:.2f} seconds.")
    print(f"[+] Exit code: {result.get('exit_code')}")

    if result.get("stderr"):
        print("[!] Stderr warnings/messages:")
        for line in result["stderr"].strip().splitlines()[:15]:
            print(f"    {line}")

    for cell in result.get("cells", []):
        stdout = cell.get("stdout", "").strip()
        if stdout:
            print("\n=== COLAB EXECUTION LOGS ===")
            for line in stdout.splitlines():
                if not line.startswith("ARTIFACT_BASE64"):
                    print(line)
            print("============================\n")

    artifact_files = result.get("artifact_files", [])
    print(f"[+] Downloaded artifact files: {artifact_files}")

    onnx_target = os.path.join(output_dir, "narrative_nano_5_8m.onnx")
    if os.path.isfile(onnx_target):
        size_bytes = os.path.getsize(onnx_target)
        print(f"[SUCCESS] ONNX Model Verified: {onnx_target}")
        print(f"  Size: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    else:
        print(f"[!] Warning: {onnx_target} not found directly. Checking extracted artifacts...")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".onnx"):
                    full_p = os.path.join(root, file)
                    print(f"  Found ONNX file: {full_p} ({os.path.getsize(full_p):,} bytes)")

if __name__ == "__main__":
    main()
