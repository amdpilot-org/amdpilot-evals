#!/usr/bin/env python3
"""Test harness for SGLang #19935 — FP8 MLA decode assertion fix.

Phase 1 (fast): Source code verification — checks that all 4 mla_decode_fwd
call sites in aiter_backend.py have the k_scale fallback applied.

Phase 2 (optional, slow): Full server test — starts the server with FP8 KV
cache and verifies it doesn't crash. Only runs if FULL_VERIFY=1.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

AITER_BACKEND = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"
MODEL_PATH = "/models/Kimi-K2.5"
PORT = 9001
SERVER_TIMEOUT = 2400
REQUEST_TIMEOUT = 120


def verify_source_code():
    """Check that the fix was applied to all 4 mla_decode_fwd call sites."""
    if not os.path.isfile(AITER_BACKEND):
        print(f"ERROR: {AITER_BACKEND} not found")
        return 0

    with open(AITER_BACKEND) as f:
        code = f.read()

    fixed_sites = 0
    total_sites = 0

    for match in re.finditer(r'mla_decode_fwd\s*\(', code):
        total_sites += 1
        start = match.start()
        region = code[max(0, start - 500):start + 800]

        has_fallback = any(p in region for p in [
            "self.k_scale if layer.k_scale is None",
            "layer.k_scale if layer.k_scale is not None else self.k_scale",
            "self.k_scale if not layer.k_scale",
            "layer.k_scale or self.k_scale",
            "k_scale = self.k_scale" and "layer.k_scale",
            "kv_scale = self.k_scale",
            "_scale = layer.k_scale if layer.k_scale",
            "layer.k_scale is None",
        ])

        if has_fallback:
            fixed_sites += 1

    if total_sites == 0:
        print("WARNING: No mla_decode_fwd call sites found in aiter_backend.py")
        has_none_check = "layer.k_scale is None" in code or "k_scale is None" in code
        has_self_k_scale = "self.k_scale" in code
        if has_none_check and has_self_k_scale:
            print("Found k_scale None-check and self.k_scale reference — fix likely applied")
            return 75
        return 0

    print(f"mla_decode_fwd call sites: {total_sites} found, {fixed_sites} fixed")

    if fixed_sites == total_sites and total_sites >= 4:
        return 100
    elif fixed_sites == total_sites and total_sites > 0:
        return 90
    elif fixed_sites > 0:
        return int(25 * fixed_sites)
    return 0


def verify_server():
    """Optional: start the actual server and verify it doesn't crash."""
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH} — skipping server test")
        return None

    env = os.environ.copy()
    env["SGLANG_AITER_MLA_PERSIST"] = "1"

    server_cmd = [
        "/opt/venv/bin/python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tensor-parallel-size", "4",
        "--trust-remote-code",
        "--chunked-prefill-size", "131072",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--disable-radix-cache",
        "--mem-fraction-static", "0.8",
        "--max-running-requests", "64",
        "--kv-cache-dtype", "fp8_e4m3",
        "--attention-backend", "aiter",
    ]

    print(f"\n=== Full server verification (timeout {SERVER_TIMEOUT}s) ===")
    print(f"Starting server on port {PORT}...")
    proc = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        start = time.time()
        while time.time() - start < SERVER_TIMEOUT:
            try:
                req = urllib.request.Request(f"http://localhost:{PORT}/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = int(time.time() - start)
                        print(f"Server healthy after {elapsed}s")
                        payload = json.dumps({
                            "text": "What is 2 + 2?",
                            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                        }).encode()
                        req = urllib.request.Request(
                            f"http://localhost:{PORT}/generate",
                            data=payload,
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                            resp_data = json.loads(r.read().decode())
                        text = resp_data.get("text", "")
                        print(f"Response: {text[:200]}")
                        return 100 if text.strip() else 50
            except Exception:
                pass

            ret = proc.poll()
            if ret is not None:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                if "assertion" in stdout.lower():
                    print("ASSERTION ERROR — fix did not resolve the crash")
                    return 0
                print(f"Server exited with code {ret}")
                return 0
            time.sleep(10)

        print("Server startup timed out")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    print("=== Phase 1: Source code verification ===")
    code_score = verify_source_code()
    print(f"Code verification score: {code_score}")

    if os.environ.get("FULL_VERIFY") == "1" and code_score >= 75:
        server_score = verify_server()
        if server_score is not None:
            print(f"Server verification score: {server_score}")
            final = min(code_score, server_score)
            print(f"SCORE: {final}")
            return

    print(f"SCORE: {code_score}")


if __name__ == "__main__":
    main()
