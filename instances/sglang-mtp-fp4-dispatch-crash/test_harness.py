#!/usr/bin/env python3
"""
Test harness for sglang-mtp-fp4-dispatch-crash.

Validates that MTP layers function correctly when the main model uses
a different quantization format by running the server end-to-end and
verifying inference completes without dispatch crashes.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model path for behavioral testing (volume-mounted from host)
MODEL_CACHE = "/root/.cache/huggingface/"

# Error pattern that indicates the bug is NOT fixed
CRASH_PATTERN = re.compile(r"Unsupported kernel config for moe heuristic dispatch")

# Server config
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 30000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


def find_mxfp4_model():
    """Look for an MXFP4/FP8 DeepSeek model in the HuggingFace cache."""
    if not os.path.isdir(MODEL_CACHE):
        return None

    for root, dirs, files in os.walk(MODEL_CACHE):
        depth = root.replace(MODEL_CACHE, "").count(os.sep)
        if depth > 3:
            continue

        for d in dirs:
            d_lower = d.lower()
            if "deepseek" in d_lower:
                full_path = os.path.join(root, d)
                config_path = os.path.join(full_path, "config.json")
                if os.path.isfile(config_path):
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        quant_config = config.get("quantization_config", {})
                        if quant_config.get("quant_method") in [
                            "fp4", "mxfp4", "fbgemm_fp8",
                        ]:
                            return full_path
                    except (json.JSONDecodeError, KeyError):
                        pass

    return None


def start_server(model_path):
    """Start sglang server with MTP + MoE dispatch."""
    env = os.environ.copy()
    env["SGLANG_MOE_DISPATCH"] = "mori"

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(SERVER_PORT),
        "--host", SERVER_HOST,
        "--tp", "2",
        "--trust-remote-code",
        "--speculative-algorithm", "EAGLE",
    ]

    print(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True,
    )
    return proc


def wait_for_server(proc, timeout=300):
    """Wait for server to start. Returns (success, logs)."""
    logs = []
    start_time = time.time()

    while time.time() - start_time < timeout:
        ret = proc.poll()
        if ret is not None:
            remaining = proc.stdout.read()
            if remaining:
                logs.append(remaining)
            return False, "".join(logs)

        try:
            line = proc.stdout.readline()
            if line:
                logs.append(line)
                print(f"  SERVER: {line.rstrip()}")
                if CRASH_PATTERN.search(line):
                    return False, "".join(logs)
        except Exception:
            pass

        try:
            req = urllib.request.Request(f"{SERVER_URL}/health")
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                return True, "".join(logs)
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass

        time.sleep(2)

    return False, "".join(logs)


def send_inference_request():
    """Send a simple inference request to verify the server works."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        return True, content
    except Exception as e:
        return False, str(e)


def kill_server(proc):
    """Kill the server process and its children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("sglang-mtp-fp4-dispatch-crash test harness")
    print("=" * 60)

    model_path = find_mxfp4_model()

    if model_path is None:
        print("\n--- No MXFP4/FP8 DeepSeek model found ---")
        print(f"  Looked in: {MODEL_CACHE}")
        check("Model available for behavioral test", False,
              "No MXFP4 model available — behavioral verification required")
        check("Server starts without MoE dispatch crash", False,
              "Cannot test without model")
        check("Inference produces valid output", False,
              "Cannot test without model")
        print()
        score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
        print(f"Results: {checks_passed}/{checks_total}")
        print(f"SCORE: {score:.1f}")
        sys.exit(1)

    check("Model available for behavioral test", True)
    print(f"  Using model: {model_path}")

    # Start server with MTP + MORI dispatch
    print("\n--- Server startup test ---")
    proc = start_server(model_path)
    try:
        server_ready, logs = wait_for_server(proc, timeout=300)

        crash_found = bool(CRASH_PATTERN.search(logs))

        if crash_found:
            check("Server starts without MoE dispatch crash", False,
                  "Crashed with 'Unsupported kernel config for moe heuristic dispatch'")
            check("Inference produces valid output", False,
                  "Server crashed during startup")
        elif not server_ready:
            check("Server starts without MoE dispatch crash", False,
                  "Server did not become ready within timeout")
            check("Inference produces valid output", False,
                  "Server not ready")
        else:
            check("Server starts without MoE dispatch crash", True)

            # Send inference request
            print("\n--- Inference test ---")
            success, response = send_inference_request()
            check("Inference produces valid output", success,
                  f"Error: {response}" if not success else "")
            if success:
                print(f"  Response: {response[:200]}")

            # Send a second request to verify consistency
            success2, response2 = send_inference_request()
            check("Second inference request succeeds", success2,
                  f"Error: {response2}" if not success2 else "")

    finally:
        kill_server(proc)

    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total}")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
