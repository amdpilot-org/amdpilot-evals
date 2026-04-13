#!/usr/bin/env python3
"""Test harness for sglang-tp4-accuracy-cross-repo.

Starts the server with TP=4 and the aiter attention backend, sends
inference requests, and checks that outputs are coherent and factually
correct. The bug causes subtle accuracy degradation at TP=4 due to
incorrect MLA kernel configuration for the 32-head case.
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

# Find model
_MODEL_DIRS = [
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8",
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct",
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
    print("SGLang TP=4 Accuracy Cross-Repo Fix Test")
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

    log_file = open("/tmp/sglang_tp4_test.log", "w")
    server = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("  Waiting for server (model loading ~15-30 min)...")
        if not _wait_for_server():
            check("Server starts at TP=4", False,
                  "Server did not become ready")
            print(f"\nSCORE: 0.0")
            return

        check("Server starts at TP=4", True)

        # Warmup
        _send_request("Hello")

        # --- Factual accuracy checks ---
        # The bug causes accuracy degradation at TP=4 due to incorrect
        # MLA kernel configuration. Test with a mix of simple facts and
        # multi-step reasoning that amplifies attention errors.
        print("\n--- Factual accuracy checks ---")
        accuracy_prompts = [
            ("What is 2 + 2? Answer with just the number.", "4"),
            ("What is the capital of Japan? Answer in one word.", "tokyo"),
            ("Is the Earth round? Answer yes or no.", "yes"),
            ("What color is the sky on a clear day? One word.", "blue"),
            ("How many legs does a cat have? Just the number.", "4"),
            ("What is 15 minus 7? Answer with just the number.", "8"),
            ("What is the chemical symbol for water? Answer in letters only.", "h2o"),
            ("How many days are in a week? Just the number.", "7"),
        ]

        correct = 0
        for prompt, expected in accuracy_prompts:
            response = _send_request(prompt, max_tokens=32)
            if response and expected.lower() in response.lower():
                correct += 1
            else:
                print(f"    MISS: expected '{expected}' in: "
                      f"{(response or 'None')[:80]}")

        check("Factual accuracy at TP=4",
              correct >= 6,
              f"Only {correct}/{len(accuracy_prompts)} correct "
              f"(expected >= 6)")

        # --- Multi-step reasoning ---
        # Degraded attention heads compound errors in reasoning chains.
        print("\n--- Reasoning checks ---")
        reasoning_prompts = [
            ("If Alice has 5 apples and gives 2 to Bob, how many does "
             "Alice have left? Answer with just the number.", "3"),
            ("A rectangle has length 4 and width 3. What is its area? "
             "Just the number.", "12"),
            ("If today is Wednesday, what day was it 2 days ago? "
             "One word answer.", "monday"),
        ]

        reasoning_correct = 0
        for prompt, expected in reasoning_prompts:
            response = _send_request(prompt, max_tokens=32)
            if response and expected.lower() in response.lower():
                reasoning_correct += 1
            else:
                print(f"    MISS: expected '{expected}' in: "
                      f"{(response or 'None')[:80]}")

        check("Multi-step reasoning at TP=4",
              reasoning_correct >= 2,
              f"Only {reasoning_correct}/{len(reasoning_prompts)} correct "
              f"(expected >= 2)")

        # --- Coherence checks ---
        # Longer generation to amplify degradation effects.
        print("\n--- Coherence checks ---")
        coherence_prompts = [
            "Explain what water is in two or three sentences.",
            "List five common fruits and one sentence about each.",
            "Describe the process of photosynthesis briefly.",
            "What are three differences between cats and dogs?",
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

        check("Output coherence at TP=4",
              coherent >= 3,
              f"Only {coherent}/{len(coherence_prompts)} coherent responses "
              f"(expected >= 3)")

        # --- Consistency check ---
        # Same prompt 3 times with temp=0 should give consistent results.
        print("\n--- Consistency check ---")
        consistency_prompt = "What is the largest planet in our solar system? One word."
        responses = []
        for _ in range(3):
            r = _send_request(consistency_prompt, max_tokens=16)
            if r:
                responses.append(r.strip().lower())

        if len(responses) == 3:
            has_jupiter = sum(1 for r in responses if "jupiter" in r)
            all_same = len(set(responses)) == 1
            check("Consistency at TP=4",
                  has_jupiter >= 2 and all_same,
                  f"Responses: {responses}")
        else:
            check("Consistency at TP=4", False,
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
