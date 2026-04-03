#!/usr/bin/env python3
"""Test harness for sglang#20691: Kimi K2 crashes on ROCm with fused decode MLA.

Re-formulated with v0.4.0 structural fixes:
- Safe kill pattern (lesson #4)
- Correct Python path (lesson #6)
- No PYTHONPATH blanking
- Server process isolation via preexec_fn + os.setsid
"""
import json
import os
import signal
import subprocess
import sys
import time

import requests

PYTHON = "/opt/venv/bin/python3"
MODEL = "fxmarty/moonshotai_Kimi-K2-Instruct-0905-2-layers"
PORT = 30100
BASE_URL = f"http://127.0.0.1:{PORT}"

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}" + (f": {detail}" if detail else ""))


def kill_server():
    """Kill any running sglang server processes using SAFE pattern (lesson #4).

    Uses pgrep -f 'python3 -m sglang' | xargs -r kill -9
    instead of kill -9 $(pgrep -f sglang) which would match the agent process.
    """
    subprocess.run(
        'pgrep -f "python3 -m sglang" | xargs -r kill -9',
        shell=True, capture_output=True
    )
    time.sleep(3)


def wait_for_server(timeout=1800):
    """Wait for the server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"  Server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def main():
    global checks_passed, checks_total

    print("=" * 60)
    print("Test Harness: sglang#20691 — Kimi K2 fused decode MLA crash")
    print("=" * 60)

    # === Cleanup: kill any leftover servers ===
    print("\nKilling existing sglang processes...")
    kill_server()

    # === Test 1: Server starts with default fused decode MLA (no workaround env var) ===
    print("\n=== Test 1: Start server with default fused decode MLA ===")

    env = os.environ.copy()
    # Ensure SGLANG_ROCM_FUSED_DECODE_MLA is NOT set (default = enabled)
    env.pop("SGLANG_ROCM_FUSED_DECODE_MLA", None)

    # Match the issue's reproduction command exactly.
    # DO NOT add --disable-cuda-graph — the bug triggers during CUDA graph capture.
    # DO NOT remove --decode-attention-backend triton or --prefill-attention-backend aiter
    # — these match the issue's reported config.
    server_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--tensor-parallel-size", "8",
        "--trust-remote-code",
        "--decode-attention-backend", "triton",
        "--prefill-attention-backend", "aiter",
        "--port", str(PORT),
        "--host", "0.0.0.0",
    ]

    print(f"Starting server: {' '.join(server_cmd)}")
    log_file = open("/workspace/server_log.txt", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )

    try:
        server_ready = wait_for_server(timeout=1800)
        check("Server starts successfully with fused decode MLA enabled", server_ready,
              "Server failed to start — likely the AttributeError/TypeError crash")

        # === Test 2: Server responds to a generate request ===
        if server_ready:
            print("\n=== Test 2: Send a generate request ===")
            try:
                resp = requests.post(
                    f"{BASE_URL}/v1/completions",
                    json={
                        "model": MODEL,
                        "prompt": "Hello, how are you?",
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                    timeout=120,
                )
                result = resp.json()
                has_text = (
                    "choices" in result
                    and len(result["choices"]) > 0
                    and len(result["choices"][0].get("text", "")) > 0
                )
                check("Server returns a valid completion response", has_text,
                      f"Response: {json.dumps(result, indent=2)[:500]}")
            except Exception as e:
                check("Server returns a valid completion response", False, str(e))
        else:
            print("\n=== Test 2: Skipped (server not ready) ===")
            check("Server returns a valid completion response", False, "Server not ready")

    finally:
        # === Cleanup: kill server process group ===
        print("\nCleaning up server process...")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        server_proc.kill()
        server_proc.wait()
        log_file.close()

    # === Test 3: Check logs for known error patterns ===
    print("\n=== Test 3: Check server logs for crash errors ===")
    try:
        with open("/workspace/server_log.txt", "r") as f:
            log_content = f.read()

        has_attr_error = (
            "AttributeError: 'HybridAttnBackend' object has no attribute 'forward_metadata'"
            in log_content
        )
        has_type_error = (
            "cannot unpack non-iterable ForwardMetada" in log_content
            or "cannot unpack non-iterable ForwardMetadata" in log_content
        )
        has_sigquit = "Received sigquit from a child process" in log_content

        check("No AttributeError on forward_metadata", not has_attr_error,
              "Found AttributeError for forward_metadata in logs")
        check("No TypeError on ForwardMetadata unpacking", not has_type_error,
              "Found TypeError for ForwardMetadata unpacking in logs")
        check("No sigquit crash signal", not has_sigquit,
              "Found sigquit crash signal in logs")
    except Exception as e:
        check("No AttributeError on forward_metadata", False, str(e))
        check("No TypeError on ForwardMetadata unpacking", False, str(e))
        check("No sigquit crash signal", False, str(e))

    # Final cleanup
    kill_server()

    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"\nResults: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
