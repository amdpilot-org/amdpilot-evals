#!/usr/bin/env python3
"""Test harness for sglang MLA decode NaN-on-padded-rows issue.

Starts the server with speculative decoding and the aiter attention backend,
sends inference requests, and checks whether outputs are coherent (no NaN
or garbage from unwritten padded positions in the output buffer).
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

_PY = "/opt/venv/bin/python3"
_PORT = 30000
_URL = f"http://localhost:{_PORT}"
_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b"
_TIMEOUT_STARTUP = 2400
_TIMEOUT_REQUEST = 300

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def _kill_existing():
    subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(2)


def _wait_for_server():
    start = time.time()
    while time.time() - start < _TIMEOUT_STARTUP:
        try:
            req = urllib.request.Request(f"{_URL}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(10)
    return False


def _send_request(prompt):
    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT_REQUEST)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("SGLang MLA Decode NaN-Padded Test")
    print("=" * 60)

    # --- Server inference test ---
    print("\n--- Server inference sanity ---")

    _kill_existing()

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    server_cmd = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "2",
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--attention-backend", "aiter",
        "--disable-radix-cache",
        "--mem-fraction-static", "0.8",
        "--port", str(_PORT),
    ]

    log_file = open("/tmp/sglang_nan_test.log", "w")
    server = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("  Waiting for server (model loading ~15-30 min)...")
        if not _wait_for_server():
            check("Server starts and produces valid output", False,
                  "server did not become ready")
        else:
            # Send multiple requests to increase probability of hitting the
            # non-deterministic NaN bug. The NaN depends on GPU memory state
            # and manifests intermittently during speculative decode verify.
            test_prompts = [
                f"What is {i} + {i}?" for i in range(1, 11)
            ]
            good = 0
            nan_or_empty = 0
            for prompt in test_prompts:
                response = _send_request(prompt)
                if response and len(response.strip()) > 3:
                    good += 1
                else:
                    nan_or_empty += 1

            check("Server starts and produces valid output",
                  good >= 7,
                  f"only {good}/{len(test_prompts)} valid responses "
                  f"({nan_or_empty} empty/corrupted)")

    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log_file.close()
        _kill_existing()

    # --- Summary ---
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
