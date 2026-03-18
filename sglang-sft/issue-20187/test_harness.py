#!/usr/bin/env python3
"""Test harness for SGLang #20187 — FP8 prefill + radix cache integration.

Starts the server with FP8 prefill + radix cache (enabled by default) for
DeepSeek-R1-MXFP4, then verifies it starts and serves requests correctly.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

MODEL_PATH = "/models/DeepSeek-R1-MXFP4-Preview"
PORT = 9000
SERVER_TIMEOUT = 600
REQUEST_TIMEOUT = 180


def wait_for_server(port, timeout):
    start = time.time()
    while time.time() - start < timeout:
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def send_test_request(port):
    url = f"http://localhost:{port}/generate"
    payload = json.dumps({
        "text": "What is 2 + 2? Answer briefly.",
        "sampling_params": {"temperature": 0, "max_new_tokens": 64},
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}", file=sys.stderr)
        print("SCORE: 0")
        return

    env = os.environ.copy()
    env["SGLANG_AITER_FP8_PREFILL_ATTN"] = "1"

    server_cmd = [
        "/opt/venv/bin/python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tensor-parallel-size", "8",
        "--trust-remote-code",
        "--chunked-prefill-size", "131072",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--mem-fraction-static", "0.8",
        "--max-running-requests", "64",
        "--kv-cache-dtype", "fp8_e4m3",
        "--attention-backend", "aiter",
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ]

    print(f"Starting server: {' '.join(server_cmd)}")
    proc = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        print(f"Waiting for server to start (timeout {SERVER_TIMEOUT}s)...")
        if not wait_for_server(PORT, SERVER_TIMEOUT):
            print("Server failed to start within timeout", file=sys.stderr)
            print("SCORE: 0")
            return

        print("Server is up! Sending test request...")
        resp = send_test_request(PORT)
        text = resp.get("text", "")
        print(f"Response: {text[:200]}")

        if text.strip():
            print("SCORE: 100")
        else:
            print("Empty response from server")
            print("SCORE: 50")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("SCORE: 0")
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
