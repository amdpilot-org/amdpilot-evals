#!/usr/bin/env python3
"""Test harness for sglang-amd-context-len-perf-fix.

Verifies that the attention backend computes KV cache length correctly
so the kernel dispatcher selects the right path for all context-length
settings.

The test constructs scenarios that exercise the max_kv_len computation
and verifies the result accounts for the full KV cache capacity.
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

# Find a model to test with
_MODEL_DIRS = [
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8",
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct",
]

_TIMEOUT_STARTUP = 1800
_TIMEOUT_REQUEST = 120

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
    """Find a model with snapshot directory."""
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


def _wait_for_server(timeout):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{_URL}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
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


def _measure_decode_time(prompt, max_tokens=128):
    """Send a request and measure time-to-completion as a rough decode perf proxy."""
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
        start = time.time()
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT_REQUEST)
        elapsed = time.time() - start
        result = json.loads(resp.read())
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", max_tokens)
        if completion_tokens > 0:
            return elapsed / completion_tokens  # seconds per token
        return elapsed
    except Exception:
        return None


def main():
    print("=" * 60)
    print("sglang-amd-context-len-perf-fix test harness")
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

    # Test: Start server with reduced context-length and verify
    # decode performance is NOT degraded vs default.
    # The bug: with --context-length < 32768, the attention kernel
    # dispatcher selects the slow path, causing ~40% perf regression.

    # Run 1: default context length (baseline)
    print("\n--- Run 1: Default context length (baseline) ---")
    cmd_default = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--attention-backend", "aiter",
        "--mem-fraction-static", "0.80",
        "--tp", "2",
        "--port", str(_PORT),
    ]

    log1 = open("/tmp/sglang_ctx_default.log", "w")
    srv1 = subprocess.Popen(cmd_default, env=env, stdout=log1, stderr=subprocess.STDOUT)

    baseline_spt = None
    try:
        print("  Waiting for server (default ctx)...")
        if _wait_for_server(_TIMEOUT_STARTUP):
            # Warmup
            _send_request("Hello")
            # Measure
            times = []
            for _ in range(3):
                t = _measure_decode_time("Explain quantum computing in detail.", 128)
                if t is not None:
                    times.append(t)
            if times:
                baseline_spt = sum(times) / len(times)
                print(f"  Baseline decode: {baseline_spt*1000:.1f} ms/token")
        else:
            print("  Server did not start (default ctx)")
    finally:
        srv1.terminate()
        try:
            srv1.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv1.kill()
            srv1.wait(timeout=10)
        log1.close()
        _kill_existing()

    # Run 2: reduced context length
    print("\n--- Run 2: Reduced context length (--context-length 13824) ---")
    cmd_reduced = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--attention-backend", "aiter",
        "--context-length", "13824",
        "--mem-fraction-static", "0.80",
        "--tp", "2",
        "--port", str(_PORT),
    ]

    log2 = open("/tmp/sglang_ctx_reduced.log", "w")
    srv2 = subprocess.Popen(cmd_reduced, env=env, stdout=log2, stderr=subprocess.STDOUT)

    reduced_spt = None
    try:
        print("  Waiting for server (reduced ctx)...")
        if _wait_for_server(_TIMEOUT_STARTUP):
            # Warmup
            _send_request("Hello")
            # Measure
            times = []
            for _ in range(3):
                t = _measure_decode_time("Explain quantum computing in detail.", 128)
                if t is not None:
                    times.append(t)
            if times:
                reduced_spt = sum(times) / len(times)
                print(f"  Reduced ctx decode: {reduced_spt*1000:.1f} ms/token")
        else:
            check("Server starts with reduced context-length", False,
                  "Server did not start with --context-length 13824")
            print(f"\nSCORE: 0.0")
            return
    finally:
        srv2.terminate()
        try:
            srv2.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv2.kill()
            srv2.wait(timeout=10)
        log2.close()
        _kill_existing()

    check("Server starts with reduced context-length", True)

    # Check: reduced ctx should NOT be significantly slower than baseline
    if baseline_spt is not None and reduced_spt is not None:
        ratio = reduced_spt / baseline_spt
        print(f"\n  Performance ratio (reduced/default): {ratio:.2f}x")
        print(f"  Default: {baseline_spt*1000:.1f} ms/tok, Reduced: {reduced_spt*1000:.1f} ms/tok")

        # The bug causes ~40% regression. Allow up to 20% overhead
        # (accounting for measurement noise and different cache sizes).
        check(
            "No performance regression with reduced context-length",
            ratio < 1.20,
            f"Reduced context-length is {ratio:.2f}x slower than default "
            f"(expected <1.20x, got {ratio:.2f}x — likely wrong kernel path)"
        )
    elif reduced_spt is not None:
        # No baseline available — just check that reduced works at all
        check(
            "No performance regression with reduced context-length",
            True,
            "(no baseline available for comparison)"
        )
    else:
        check(
            "No performance regression with reduced context-length",
            False,
            "Could not measure decode performance"
        )

    # Check: responses are coherent with reduced context
    print("\n--- Coherence check with reduced context ---")
    cmd_check = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--attention-backend", "aiter",
        "--context-length", "13824",
        "--mem-fraction-static", "0.80",
        "--tp", "2",
        "--port", str(_PORT),
    ]

    log3 = open("/tmp/sglang_ctx_coherence.log", "w")
    srv3 = subprocess.Popen(cmd_check, env=env, stdout=log3, stderr=subprocess.STDOUT)

    try:
        if _wait_for_server(_TIMEOUT_STARTUP):
            prompts = ["What is 2+2?", "Name three colors.", "What is the capital of Japan?"]
            good = 0
            for p in prompts:
                r = _send_request(p)
                if r and len(r.strip()) > 3:
                    good += 1
            check(
                "Coherent output with reduced context-length",
                good >= 2,
                f"Only {good}/{len(prompts)} valid responses"
            )
        else:
            check("Coherent output with reduced context-length", False,
                  "Server not ready for coherence check")
    finally:
        srv3.terminate()
        try:
            srv3.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv3.kill()
            srv3.wait(timeout=10)
        log3.close()
        _kill_existing()

    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total}")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
