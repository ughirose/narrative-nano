import os
import sys
import json
import time
import base64
import hashlib
import mcp_server_colab_exec.server as colab_server

def main():
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "colab_run_v16_pipeline.py"))
    print(f"[*] Target Colab Pipeline: {script_path}")
    print(f"[*] Script Size: {os.path.getsize(script_path):,} bytes")

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "models"))
    os.makedirs(output_dir, exist_ok=True)
    target_onnx = os.path.join(output_dir, "narrative_nano_v16_int8.onnx")

    print("[*] Dispatching to Google Colab T4 GPU via colab_execute_file...")
    t0 = time.time()

    raw_result = colab_server.colab_execute_file(
        file_path=script_path,
        accelerator="T4",
        timeout=600
    )

    elapsed_sec = time.time() - t0
    # AFL-6: Tesla T4 rate = 1.5 CU/hour = 0.025 CU/minute
    cu_consumed = round((elapsed_sec / 3600.0) * 1.50, 4)

    print(f"[+] Colab execution finished in {elapsed_sec:.2f} seconds.")
    print(f"[+] AFL-6 CU Consumed: {cu_consumed:.4f} CU")

    result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    exit_code = result.get("exit_code")
    print(f"[+] Colab Exit Code: {exit_code}")

    if result.get("stderr"):
        print("[!] Colab Stderr:")
        for line in result["stderr"].strip().splitlines()[:20]:
            print(f"    {line}")

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    full_stdout = ""
    for cell in result.get("cells", []):
        full_stdout += cell.get("stdout", "") + "\n"

    # Save complete stdout to log file
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "v16_colab_run.log"))
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(full_stdout)
    print(f"[+] Saved execution log to: {log_file}")

    # Print logs without base64 dump
    print("\n=== COLAB EXECUTION LOGS ===")
    for line in full_stdout.splitlines():
        if "ARTIFACT_V16_B64" not in line and len(line) < 300:
            try:
                print(line)
            except Exception:
                print(line.encode('ascii', errors='replace').decode('ascii'))
    print("============================\n")

    if exit_code != 0:
        print(f"[ERROR] Colab pipeline failed with exit code {exit_code}")
        sys.exit(1)

    # Extract base64 artifact
    start_tag = "ARTIFACT_V16_B64_START"
    end_tag = "ARTIFACT_V16_B64_END"
    if start_tag in full_stdout and end_tag in full_stdout:
        b64_data = full_stdout.split(start_tag)[1].split(end_tag)[0].strip()
        model_bytes = base64.b64decode(b64_data)
        with open(target_onnx, "wb") as f:
            f.write(model_bytes)
        
        sha = hashlib.sha256(model_bytes).hexdigest()
        size_mb = len(model_bytes) / (1024 * 1024)
        print(f"[SUCCESS] Retrieved Model: {target_onnx}")
        print(f"  Size: {size_mb:.2f} MB ({len(model_bytes):,} bytes)")
        print(f"  SHA-256: {sha}")

        # Also update plotailor/dist/models/narrative_nano_latest.onnx
        plotailor_target = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plotailor", "dist", "models", "narrative_nano_latest.onnx"))
        os.makedirs(os.path.dirname(plotailor_target), exist_ok=True)
        with open(plotailor_target, "wb") as f:
            f.write(model_bytes)
        print(f"[SUCCESS] Updated Plotailor Model: {plotailor_target}")
    else:
        print("[ERROR] Could not find base64 artifact tags in Colab output!")
        sys.exit(1)

if __name__ == "__main__":
    main()
