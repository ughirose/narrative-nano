#!/usr/bin/env python3
"""
Orchestrates Google Colab execution of RUN-018 & RUN-019 pipeline,
monitors real-time progress via WebSocket with automatic ping keepalive,
collects logs, downloads generated INT8 ONNX model, and verifies integrity.
"""
import os
import sys
import re
import json
import time
import uuid
import websocket

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_write(stream, text):
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        try:
            encoding = stream.encoding or "utf-8"
            cleaned = text.encode(encoding, errors="replace").decode(encoding)
            stream.write(cleaned)
            stream.flush()
        except Exception:
            pass
    except Exception:
        pass

from mcp_server_colab_exec.colab_runtime import (
    get_credentials,
    allocate_runtime,
    start_keepalive,
    create_session,
    unassign_runtime,
    propagate_credentials,
    EPHEMERAL_AUTH_TYPES,
    _make_colab_input_reply
)
from download_model_helper import download_file

def execute_on_colab(code: str, accelerator: str = "T4", timeout: int = 1500):
    creds = get_credentials()
    access_token = creds.token

    print(f"[*] Allocating Colab {accelerator} runtime...")
    assignment = allocate_runtime(access_token, accelerator)
    endpoint = assignment["endpoint"]
    proxy_url = assignment["proxy_url"]
    proxy_token = assignment["proxy_token"]

    print(f"[*] Endpoint: {endpoint}")
    print(f"[*] Starting HTTP keepalive loop...")
    stop_event = start_keepalive(access_token, endpoint)

    try:
        print("[*] Creating Jupyter kernel session...")
        kernel_id = create_session(proxy_url, proxy_token)
        print(f"[*] Kernel ID: {kernel_id}")

        execute_session_id = uuid.uuid4().hex
        ws_url = proxy_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels?session_id={execute_session_id}"

        print(f"[*] Connecting WebSocket to {ws_url[:60]}...")
        ws = websocket.create_connection(
            ws_url,
            header=[
                f"X-Colab-Runtime-Proxy-Token: {proxy_token}",
                "X-Colab-Client-Agent: vscode",
            ],
            timeout=15, # 15s timeout to trigger ws.ping keepalive
        )

        msg_id = uuid.uuid4().hex
        execute_msg = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "colab-exec",
                "session": execute_session_id,
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }

        print("[*] Sending execute_request to kernel...")
        ws.send(json.dumps(execute_msg))
        print("[*] Pipeline dispatched! Streaming real-time outputs:\n" + "-" * 70)

        stdout_chunks = []
        stderr_chunks = []
        had_error = False
        saw_idle = False
        deadline = time.time() + timeout
        last_ping = time.time()

        while time.time() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                # Socket idle - ping server to prevent proxy disconnection
                try:
                    ws.ping()
                except Exception as pe:
                    sys.stderr.write(f"\n[!] Ping error: {pe}\n")
                continue
            except websocket.WebSocketConnectionClosedException as ce:
                sys.stderr.write(f"\n[!] WebSocket connection closed: {ce}\n")
                break
            except Exception as e:
                sys.stderr.write(f"\n[!] WebSocket read error: {e}\n")
                break

            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})

            # Colab Auth challenge handling
            if msg_type == "colab_request":
                metadata = msg.get("metadata", {})
                request_type = metadata.get("colab_request_type")
                colab_msg_id = metadata.get("colab_msg_id")
                auth_type = (
                    content.get("request", {}).get("authType", "")
                    if isinstance(content, dict) else ""
                )
                auth_type = str(auth_type).lower()
                error_text = None
                if request_type == "request_auth" and colab_msg_id is not None:
                    if auth_type in EPHEMERAL_AUTH_TYPES:
                        try:
                            dry = propagate_credentials(access_token, endpoint, auth_type, dry_run=True)
                            if dry.get("success"):
                                propagate_credentials(access_token, endpoint, auth_type, dry_run=False)
                            else:
                                error_text = f"dry run: {dry}"
                        except Exception as ex:
                            error_text = str(ex)
                    else:
                        error_text = f"unsupported auth: {auth_type}"
                    reply = _make_colab_input_reply(execute_session_id, colab_msg_id, error_text)
                    ws.send(json.dumps(reply))
                continue

            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            if parent_msg_id != msg_id:
                continue

            if msg_type == "stream":
                stream_name = content.get("name", "stdout")
                text = content.get("text", "")
                if stream_name == "stdout":
                    safe_write(sys.stdout, text)
                    stdout_chunks.append(text)
                else:
                    safe_write(sys.stderr, text)
                    stderr_chunks.append(text)

            elif msg_type == "error":
                had_error = True
                ename = content.get("ename", "Error")
                evalue = content.get("evalue", "")
                traceback_lines = content.get("traceback", [])
                err_str = f"\n{ename}: {evalue}\n" + "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l) for l in traceback_lines)
                safe_write(sys.stderr, err_str + "\n")
                stderr_chunks.append(err_str)

            elif msg_type == "status":
                state = content.get("execution_state")
                if state == "idle":
                    saw_idle = True
                    break

        ws.close()
        full_stdout = "".join(stdout_chunks)
        full_stderr = "".join(stderr_chunks)
        exit_code = 1 if had_error or not saw_idle else 0
        return full_stdout, full_stderr, exit_code

    finally:
        stop_event.set()
        unassign_runtime(access_token, endpoint)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_file = os.path.join(script_dir, "colab_run018_run019_pipeline.py")
    out_model_path = os.path.abspath(
        os.path.join(script_dir, "..", "dist", "models", "narrative_nano_pro_v14_qat_int8.onnx")
    )

    print("=" * 70)
    print("[*] Worldcraft / Plotailor: RUN-018 & RUN-019 Colab GPU Pipeline")
    print("=" * 70)
    print(f"[*] Pipeline script: {pipeline_file}")
    print(f"[*] Target model:    {out_model_path}")

    with open(pipeline_file, "r", encoding="utf-8") as f:
        code = f.read()

    print(f"[*] Pipeline code size: {len(code):,} chars")

    accelerator = "T4"
    timeout_sec = 2400 # 40 minutes

    t0 = time.time()
    stdout, stderr, rc = execute_on_colab(code, accelerator=accelerator, timeout=timeout_sec)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"[*] Colab execution finished in {elapsed:.1f}s ({elapsed/60:.2f} min) with returncode: {rc}")

    # Preserve execution log
    log_file = os.path.join(script_dir, "run018_run019_colab_exec.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== EXECUTION LOG (Accelerator: {accelerator}, Elapsed: {elapsed:.1f}s, RC: {rc}) ===\n")
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    print(f"[*] Full raw execution log saved at: {log_file}")

    if rc != 0:
        print(f"[ERROR] Colab pipeline execution failed with returncode {rc}!")
        sys.exit(rc)

    # Search for AUTO_DOWNLOAD_URL
    match = re.search(r"AUTO_DOWNLOAD_URL:(.+)", stdout)
    if not match:
        print("[ERROR] AUTO_DOWNLOAD_URL not found in stdout!")
        sys.exit(1)

    url_raw = match.group(1).strip()
    print(f"[*] Raw upload response: {url_raw}")

    page_url = url_raw
    try:
        data = json.loads(url_raw)
        page_url = data.get("data", {}).get("url", url_raw)
    except Exception:
        pass

    print(f"[*] Downloading model from page: {page_url}")
    download_file(page_url, out_model_path)

    if not os.path.exists(out_model_path):
        print(f"[ERROR] Expected output file not found at {out_model_path}!")
        sys.exit(1)

    final_size_mb = os.path.getsize(out_model_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Model verified on disk: {out_model_path}")
    print(f"[SUCCESS] Size: {final_size_mb:.2f} MB")
    print(f"[SUCCESS] Pipeline RUN-018 & RUN-019 successfully executed and acquired!")

if __name__ == "__main__":
    main()
