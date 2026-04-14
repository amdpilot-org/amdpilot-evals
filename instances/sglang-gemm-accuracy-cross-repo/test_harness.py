#!/usr/bin/env python3
"""Test harness for sglang-gemm-accuracy-cross-repo.

Starts the sglang server with TP=4 and the aiter attention backend,
sends inference requests, and checks that outputs are correct and coherent.

The underlying issue is a GEMM kernel correctness bug in the FP8 blockscale
path — it produces wrong matmul results under memory pressure, which
causes the model to generate incorrect outputs.
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
_TIMEOUT_STARTUP = 2400
_TIMEOUT_REQUEST = 300

_MODEL_DIRS = [
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8",
]

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


def _find_model():
    for d in _MODEL_DIRS:
        snap = os.path.join(d, "snapshots")
        if os.path.isdir(snap):
            entries = os.listdir(snap)
            if entries:
                return os.path.join(snap, entries[0])
    return None


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


def _send_request(prompt, max_tokens=64):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
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
    print("SGLang FP8 Inference Accuracy Test")
    print("=" * 60)

    model = _find_model()
    if not model:
        check("Model available", False, "No model found in cache")
        print(f"\nSCORE: 0.0")
        return

    check("Model available", True)
    print(f"  Using model: {model}")

    _kill_existing()

    env = os.environ.copy()
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    # Start server with TP=4 and aiter backend
    print("\n--- Starting server (TP=4, aiter backend) ---")
    server_cmd = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp", "4",
        "--attention-backend", "aiter",
        "--mem-fraction-static", "0.80",
        "--port", str(_PORT),
    ]

    log_file = open("/tmp/sglang_gemm_test.log", "w")
    server = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("  Waiting for server (model loading ~15-30 min)...")
        if not _wait_for_server():
            check("Server starts", False, "Server did not become ready")
            print(f"\nSCORE: 0.0")
            return

        check("Server starts", True)

        # Warmup — also creates memory pressure from KV cache allocation
        _send_request("Hello")
        time.sleep(2)

        # --- Factual accuracy checks ---
        # Wrong GEMM outputs cause the model to produce wrong answers.
        # These are simple enough that a correct model never gets them wrong.
        print("\n--- Factual accuracy checks ---")
        accuracy_prompts = [
            ("What is 2 + 2? Answer with just the number.", "4"),
            ("What is the capital of France? Answer in one word.", "paris"),
            ("Is water wet? Answer yes or no.", "yes"),
            ("What color is grass? One word.", "green"),
            ("How many sides does a triangle have? Just the number.", "3"),
            ("What is 10 minus 3? Answer with just the number.", "7"),
            ("What is the chemical symbol for gold? Answer in letters only.", "au"),
            ("How many months are in a year? Just the number.", "12"),
        ]

        correct = 0
        for prompt, expected in accuracy_prompts:
            response = _send_request(prompt, max_tokens=32)
            if response and expected.lower() in response.lower():
                correct += 1
            else:
                print(f"    MISS: expected '{expected}' in: "
                      f"{(response or 'None')[:80]}")

        check("Factual accuracy",
              correct >= 6,
              f"Only {correct}/{len(accuracy_prompts)} correct (expected >= 6)")

        # --- Reasoning checks ---
        # GEMM errors compound through multi-step reasoning.
        print("\n--- Reasoning checks ---")
        reasoning_prompts = [
            ("If a store has 20 apples and sells 8, how many are left? "
             "Answer with just the number.", "12"),
            ("A square has side length 5. What is its perimeter? "
             "Just the number.", "20"),
            ("If it's 3pm now, what time was it 5 hours ago? "
             "Answer like '10am'.", "10"),
        ]

        reasoning_correct = 0
        for prompt, expected in reasoning_prompts:
            response = _send_request(prompt, max_tokens=32)
            if response and expected.lower() in response.lower():
                reasoning_correct += 1
            else:
                print(f"    MISS: expected '{expected}' in: "
                      f"{(response or 'None')[:80]}")

        check("Multi-step reasoning",
              reasoning_correct >= 2,
              f"Only {reasoning_correct}/{len(reasoning_prompts)} correct "
              f"(expected >= 2)")

        # --- Coherence checks ---
        # Longer outputs amplify GEMM corruption.
        print("\n--- Coherence checks ---")
        coherence_prompts = [
            "Explain what the sun is in two or three sentences.",
            "List four common vegetables and describe each in one sentence.",
            "Describe how rain forms in simple terms.",
        ]

        coherent = 0
        for prompt in coherence_prompts:
            response = _send_request(prompt, max_tokens=128)
            if response and len(response.strip()) > 20:
                words = response.split()
                if len(words) >= 8:
                    unique_ratio = len(set(w.lower() for w in words)) / len(words)
                    if unique_ratio > 0.35:
                        coherent += 1
                    else:
                        print(f"    LOW DIVERSITY: ratio={unique_ratio:.2f}")
                else:
                    print(f"    TOO SHORT: {len(words)} words")
            else:
                print(f"    EMPTY/SHORT response")

        check("Output coherence",
              coherent >= 2,
              f"Only {coherent}/{len(coherence_prompts)} coherent responses "
              f"(expected >= 2)")

        # --- Consistency check ---
        # Same prompt 3 times with temp=0 should give consistent results.
        print("\n--- Consistency check ---")
        consistency_prompt = "What is the smallest prime number? Just the number."
        responses = []
        for _ in range(3):
            r = _send_request(consistency_prompt, max_tokens=16)
            if r:
                responses.append(r.strip().lower())

        if len(responses) == 3:
            has_two = sum(1 for r in responses if "2" in r)
            all_same = len(set(responses)) == 1
            check("Consistency",
                  has_two >= 2 and all_same,
                  f"Responses: {responses}")
        else:
            check("Consistency", False,
                  f"Only got {len(responses)}/3 responses")

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
    print(f"Results: {checks_passed}/{checks_total}")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
