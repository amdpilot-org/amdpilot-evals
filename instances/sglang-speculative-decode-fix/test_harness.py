#!/usr/bin/env python3
"""Test harness for sglang speculative decode fix.

Tests:
  1. Start SGLang server with EAGLE speculative decoding + aiter backend
  2. Send multiple inference requests
  3. Check for crash or garbled output
  4. Pre-fix: server produces garbage text or crashes
  5. Post-fix: all responses are coherent
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

_PY = "/opt/venv/bin/python3"
_PORT = 30000
_URL = f"http://localhost:{_PORT}"

_MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8"

_TIMEOUT_STARTUP = 2400
_TIMEOUT_REQUEST = 300


def _find_snapshot(model_dir):
    """Find the snapshot directory inside a HF cache model dir."""
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_dir):
        return model_dir
    entries = os.listdir(snap_dir)
    if entries:
        return os.path.join(snap_dir, entries[0])
    return model_dir


_MODEL = _find_snapshot(_MODEL_DIR)

_SERVER_CMD = [
    _PY, "-m", "sglang.launch_server",
    "--model-path", _MODEL,
    "--attention-backend", "aiter",
    "--mem-fraction-static", "0.80",
    "--tp", "2",
    "--port", str(_PORT),
]

_PROMPTS = [
    "What is 2 + 2?",
    "Name three colors.",
    "What is the capital of Japan?",
    "How many days in a week?",
    "What is water made of?",
    "Name a planet in our solar system.",
    "What language is spoken in France?",
    "What is 10 times 5?",
    "Name a fruit that is red.",
    "What season comes after summer?",
]


def _kill_existing():
    subprocess.run(
        ["pkill", "-f", "sglang.launch_server"],
        capture_output=True,
    )
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
        "model": "default",
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
        content = result["choices"][0]["message"]["content"]
        return content
    except Exception:
        return None


def _is_garbled(text):
    """Check if text shows signs of garbled or corrupted output."""
    if text is None:
        return True
    if len(text.strip()) < 3:
        return True
    # Check for garbage byte patterns
    if re.search(r'[^\x20-\x7E\n\t]{5,}', text):
        return True
    # Check for excessive repetition (common garbled pattern)
    if re.search(r'(.{3,})\1{5,}', text):
        return True
    return False


def main():
    print("=" * 60)
    print("SGLang Speculative Decode Fix Test")
    print(f"  Model: {_MODEL}")
    print(f"  {len(_PROMPTS)} prompts")
    print("=" * 60)

    if not os.path.isdir(_MODEL):
        print(f"[FAIL] Model not found: {_MODEL}")
        print("SCORE: 0.0")
        return

    _kill_existing()

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    log_path = "/tmp/sglang_spec_decode_test.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        _SERVER_CMD,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("Waiting for server startup (model loading ~15-30 min)...")
        if not _wait_for_server():
            print("[FAIL] Server did not become ready")
            print("SCORE: 0.0")
            return

        print("Server ready. Sending requests...\n")

        total = len(_PROMPTS)
        good = 0
        consecutive_fails = 0

        for i, prompt in enumerate(_PROMPTS):
            print(f"  [{i+1}/{total}] {prompt}")
            response = _send_request(prompt)

            if _is_garbled(response):
                print(f"    FAIL: {repr(response)[:100]}")
                consecutive_fails += 1
            else:
                print(f"    OK: {response[:80]}")
                good += 1
                consecutive_fails = 0

            if consecutive_fails >= 3:
                print(f"\n  Short-circuit: {consecutive_fails} consecutive failures")
                break

        score = good / total * 100.0
        print(f"\n--- Results ---")
        print(f"  {good}/{total} valid responses ({score:.1f}%)")
        print(f"SCORE: {score:.1f}")

    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log_file.close()
        _kill_existing()


if __name__ == "__main__":
    main()
