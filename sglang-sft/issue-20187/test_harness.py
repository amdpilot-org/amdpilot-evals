#!/usr/bin/env python3
"""Test harness for SGLang #20187 — FP8 prefill + radix cache integration.

Phase 1 (fast): Source code verification — checks that FP8 prefill attention
was added to the radix-cache code path in aiter_backend.py.

Phase 2 (optional, slow): Full server test — starts the server with FP8
prefill + radix cache and verifies it works. Only runs if FULL_VERIFY=1.
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
MODEL_PATH = "/models/DeepSeek-R1-MXFP4-Preview"
PORT = 9001
SERVER_TIMEOUT = 2400
REQUEST_TIMEOUT = 180


def verify_source_code():
    """Check that FP8 prefill was integrated into the radix-cache path."""
    if not os.path.isfile(AITER_BACKEND):
        print(f"ERROR: {AITER_BACKEND} not found")
        return 0

    with open(AITER_BACKEND) as f:
        code = f.read()

    score = 0
    checks_passed = 0
    total_checks = 4

    has_fp8_prefill_env = "SGLANG_AITER_FP8_PREFILL_ATTN" in code
    if has_fp8_prefill_env:
        checks_passed += 1
        print("CHECK 1 PASS: FP8 prefill env var referenced")
    else:
        print("CHECK 1 FAIL: SGLANG_AITER_FP8_PREFILL_ATTN not found")

    has_fused_gemm = "fused_gemm_afp4wfp4_split_cat" in code
    if has_fused_gemm:
        checks_passed += 1
        print("CHECK 2 PASS: fused_gemm_afp4wfp4_split_cat found")
    else:
        print("CHECK 2 FAIL: fused_gemm_afp4wfp4_split_cat not found")

    has_fp8_cast = any(p in code for p in [
        "to(torch.float8_e4m3fn",
        "fp8_e4m3",
        "cast_to_fp8",
        "float8_e4m3fnuz",
    ])
    if has_fp8_cast:
        checks_passed += 1
        print("CHECK 3 PASS: FP8 casting operations found")
    else:
        print("CHECK 3 FAIL: No FP8 cast operations found")

    lines = code.split("\n")
    radix_context = False
    fp8_in_radix = False
    for i, line in enumerate(lines):
        if "radix" in line.lower() or "prefix" in line.lower():
            radix_context = True
        if radix_context and any(p in line for p in [
            "fp8", "float8", "fused_gemm", "SGLANG_AITER_FP8_PREFILL",
        ]):
            fp8_in_radix = True
            break
        if radix_context and ("def " in line and not line.strip().startswith("#")):
            radix_context = False

    if fp8_in_radix:
        checks_passed += 1
        print("CHECK 4 PASS: FP8 operations found near radix-cache code")
    else:
        print("CHECK 4 FAIL: No FP8 operations near radix-cache code paths")

    score = int(100 * checks_passed / total_checks)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    return score


def verify_server():
    """Optional: start the server with FP8 prefill + radix cache."""
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH} — skipping server test")
        return None

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
                            "text": "What is 2 + 2? Answer briefly.",
                            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
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
