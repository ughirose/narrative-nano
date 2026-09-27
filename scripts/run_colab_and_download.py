import os
import sys
import re
import json

from mcp_server_colab_exec.server import _run_on_colab
from download_model_helper import download_file

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    colab_script_path = os.path.join(script_dir, "colab_export_download.py")
    out_model_path = os.path.join(script_dir, "..", "dist", "models", "narrative_nano_v10_dialogue_int8.onnx")
    out_model_path = os.path.abspath(out_model_path)

    print(f"[*] Reading Colab export script: {colab_script_path}")
    with open(colab_script_path, "r", encoding="utf-8") as f:
        code = f.read()

    print("[*] Dispatching to Google Colab T4 runtime...")
    stdout, stderr, rc = _run_on_colab(code, accelerator="T4", timeout=300)

    print(f"[*] Colab execution finished with returncode: {rc}")
    if stderr:
        print(f"[!] Stderr (first 500 chars): {stderr[:500]}")

    match = re.search(r"AUTO_DOWNLOAD_URL:(.+)", stdout)
    if not match:
        print("[!] AUTO_DOWNLOAD_URL not found in stdout. Full stdout tail:")
        print(stdout[-1500:])
        sys.exit(1)

    url_raw = match.group(1).strip()
    print(f"[*] Raw upload response: {url_raw}")

    try:
        data = json.loads(url_raw)
        page_url = data["data"]["url"]
    except Exception as e:
        print(f"[!] Failed to parse upload json: {e}")
        page_url = url_raw

    print(f"[*] Fetching download from: {page_url}")
    download_file(page_url, out_model_path)

    print(f"[OK] Model successfully persisted at: {out_model_path}")
    print(f"[OK] Final Size: {os.path.getsize(out_model_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
