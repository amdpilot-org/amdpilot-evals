#!/usr/bin/env python3
"""Output quality test for SGLang speculative decoding."""

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
_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b"
_TIMEOUT_STARTUP = 2400  # 40 min max for model loading
_TIMEOUT_REQUEST = 300   # 5 min per request (first request may be slow)

_SERVER_CMD = [
    _PY, "-m", "sglang.launch_server",
    "--model-path", _MODEL,
    "--tp", "2",
    "--speculative-algorithm", "EAGLE",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--enable-aiter-allreduce-fusion",
    "--attention-backend", "triton",
    "--disable-radix-cache",
    "--mem-fraction-static", "0.8",
    "--reasoning-parser", "qwen3",
    "--port", str(_PORT),
]

_GARBAGE_RE = re.compile(r'(ERER|spER|\b[A-Z]{8,}\b|[\\\"=]{5,}|!{20,})')

_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and C++?",
    "Describe how a CPU works to a 10 year old.",
    "What is the capital of France and why is it important?",
    "Explain photosynthesis step by step.",
    "Write a brief history of the internet.",
    "What are three benefits of regular exercise?",
    "What causes rainbows to appear in the sky?",
    "Describe how a car engine works.",
    "What are the planets in our solar system in order?",
    "Explain how email works from sender to receiver.",
    "What is the difference between weather and climate?",
    "Describe the water cycle in detail.",
    "How does WiFi technology work?",
    "What were the main causes of World War I?",
    "Explain how vaccines work to protect against disease.",
    "What is machine learning and how does it differ from traditional programming?",
    "Describe the process of making chocolate from cacao beans.",
    "What are the main types of renewable energy sources?",
]

_NUM_ROUNDS = 2  # Both rounds must pass for score 100.0


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
        "max_tokens": 512,
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


def _is_garbage(text):
    if not text or len(text.strip()) < 10:
        return True
    if _GARBAGE_RE.search(text):
        return True
    words = text.split()
    if len(words) > 20:
        unique = len(set(w.lower() for w in words))
        if unique / len(words) < 0.1:
            return True
    return False


def _start_server():
    """Start the SGLang server and return (process, log_file)."""
    _kill_existing()
    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    log_file = open("/tmp/sglang_harness_server.log", "w")
    server = subprocess.Popen(
        _SERVER_CMD,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return server, log_file


def _stop_server(server, log_file):
    """Stop the SGLang server cleanly."""
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)
    log_file.close()
    # Extra cleanup to ensure no zombie processes
    _kill_existing()


def _run_one_round(rnd, total_rounds):
    """Start server, send all prompts, stop server. Returns round score %."""
    print(f"\n--- Round {rnd}/{total_rounds}: Independent server launch ---")
    server, log_file = _start_server()

    try:
        print("  Waiting for server (model loading may take 15-30 minutes)...")
        if not _wait_for_server():
            print("  [FAIL] Server did not become ready within timeout")
            return 0.0

        print("  Server is ready.\n")
        print(f"  Sending {len(_PROMPTS)} inference requests...")

        total = len(_PROMPTS)
        good = 0
        consecutive_failures = 0

        for i, prompt in enumerate(_PROMPTS):
            print(f"\n  [{rnd}] Request {i + 1}/{total}: {prompt[:50]}...")
            response = _send_request(prompt)

            if response is None:
                print("    ERROR: no response received (timeout or server crash)")
                consecutive_failures += 1
            elif _is_garbage(response):
                print(f"    GARBAGE: {response[:120]}...")
                consecutive_failures += 1
            else:
                print(f"    OK: {response[:120]}...")
                good += 1
                consecutive_failures = 0

            if consecutive_failures >= 2:
                print(f"\n  Short-circuiting: {consecutive_failures} consecutive failures")
                break

        round_pct = good / total * 100.0
        print(f"\n  Round {rnd}: {good}/{total} coherent ({round_pct:.1f}%)")
        return round_pct

    finally:
        print(f"  Shutting down server (round {rnd})...")
        _stop_server(server, log_file)


def main():
    print("=" * 60)
    print("SGLang Speculative Decoding Output Quality Test")
    print(f"  {len(_PROMPTS)} prompts x {_NUM_ROUNDS} independent rounds")
    print("=" * 60)

    round_scores = []
    for rnd in range(1, _NUM_ROUNDS + 1):
        pct = _run_one_round(rnd, _NUM_ROUNDS)
        round_scores.append(pct)

        if pct < 100.0:
            print(f"\n  Round {rnd} failed — skipping remaining rounds")
            break

    print(f"\n--- Results ---")
    for rnd, pct in enumerate(round_scores, 1):
        print(f"  Round {rnd}: {pct:.1f}%")

    # Score is 100.0 only if ALL rounds pass at 100%
    score = 100.0 if all(s == 100.0 for s in round_scores) and len(round_scores) == _NUM_ROUNDS else min(round_scores)
    print(f"\nSCORE: {score:.1f}")


if __name__ == "__main__":
    main()
