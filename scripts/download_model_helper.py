import urllib.request
import re
import sys
import os

def download_file(page_url, out_path):
    print(f"Fetching download page: {page_url}")
    req = urllib.request.Request(page_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')

    match = re.search(r'href="(https://tmpfiles\.org/dl/[^"]+)"', html)
    if not match:
        raise ValueError(f"Direct download link not found in page: {page_url}")

    dl_url = match.group(1)
    print(f"Direct download URL: {dl_url}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    req_dl = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_dl) as resp_dl:
        content = resp_dl.read()

    with open(out_path, "wb") as f:
        f.write(content)

    print(f"[OK] Successfully downloaded: {len(content):,} bytes to {out_path}")
    print(f"[OK] File size on disk: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_model_helper.py <page_url> <out_path>")
        sys.exit(1)
    download_file(sys.argv[1], sys.argv[2])
