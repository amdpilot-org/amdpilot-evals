#!/usr/bin/env python3
"""Test harness for sglang EAGLE3 aiter backend on non-MLA models.

Tests:
  1. Start server with non-MLA model (Llama) + EAGLE3 spec decode + aiter backend
  2. Send multiple inference requests
  3. Check for crash (AttributeError) or garbled output
  4. Pre-fix: server crashes or produces garbled output
  5. Post-fix: all responses coherent
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

# Non-MLA model (Llama) — the bug only affects non-MLA architectures
_MODEL_DIR = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct"
_DRAFT_DIR = "/root/.cache/huggingface/hub/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge"

_TIMEOUT_STARTUP = 1200
_TIMEOUT_REQUEST = 120


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
_DRAFT = _find_snapshot(_DRAFT_DIR)

_SERVER_CMD = [
    _PY, "-m", "sglang.launch_server",
    "--model-path", _MODEL,
    "--speculative-algorithm", "EAGLE3",
    "--speculative-draft-model-path", _DRAFT,
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--attention-backend", "aiter",
    "--mem-fraction-static", "0.85",
    "--tp", "4",
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
        time.sleep(5)
    return False


def _server_crashed(log_path):
    """Check if the server log shows a crash (AttributeError on MLA attrs)."""
    try:
        with open(log_path) as f:
            log = f.read()
        if "AttributeError" in log and "max_split_per_batch" in log:
            return True
        if "AttributeError" in log and "use_mla" in log:
            return True
    except Exception:
        pass
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
    print("EAGLE3 Aiter Backend Non-MLA Test")
    print(f"  Model: {_MODEL}")
    print(f"  Draft: {_DRAFT}")
    print(f"  {len(_PROMPTS)} prompts")
    print("=" * 60)

    if not os.path.isdir(_MODEL):
        print(f"[FAIL] Model not found: {_MODEL}")
        print("SCORE: 0.0")
        return

    if not os.path.isdir(_DRAFT):
        print(f"[FAIL] Draft model not found: {_DRAFT}")
        print("SCORE: 0.0")
        return

    _kill_existing()

    env = os.environ.copy()
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    log_path = "/tmp/sglang_eagle3_test.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        _SERVER_CMD,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("Waiting for server startup...")
        if not _wait_for_server():
            if _server_crashed(log_path):
                print("[FAIL] Server crashed with MLA-related AttributeError")
            else:
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
