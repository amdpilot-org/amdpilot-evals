#!/usr/bin/env python3
"""Test harness for sglang#21614: Gibberish output at high batch concurrency on MI355X.

Re-formulated with v0.4.0 structural fixes:
- Safe kill pattern (lesson #4) — no broad pgrep that kills agent
- Correct Python path (lesson #6)
- No PYTHONPATH manipulation
- Server process isolation via preexec_fn + os.setsid
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import httpx

PYTHON = "/opt/venv/bin/python3"
SERVER_PORT = 30000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"

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
    time.sleep(5)


def start_server():
    """Start the SGLang server."""
    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--tp", "4",
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--disable-radix-cache",
        "--mem-fraction-static", "0.95",
        "--max-mamba-cache-size", "128",
        "--host", "0.0.0.0",
        "--port", str(SERVER_PORT),
    ]

    print(f"Starting server: {' '.join(cmd)}")
    log_file = open("/workspace/server_log.txt", "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file


def wait_for_server(timeout=1200):
    """Wait for the server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{SERVER_URL}/v1/models", timeout=5.0)
            if r.status_code == 200 and MODEL.split("/")[-1] in r.text:
                print(f"  Server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(10)
    print(f"  Server failed to start within {timeout}s")
    return False


async def send_request(client, idx):
    """Send a single chat completion request."""
    r = await client.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_completion_tokens": 128,
            "temperature": 0.0,
        },
        timeout=180.0,
    )
    d = r.json()
    reasoning = d["choices"][0]["message"].get("reasoning_content", "") or ""
    content = d["choices"][0]["message"].get("content", "") or ""
    return idx, reasoning + content


def is_gibberish(text: str) -> bool:
    """Check if text starts with '!' or is clearly gibberish."""
    if not text.strip():
        return True
    if text.strip().startswith("!"):
        return True
    return False


async def run_concurrency_test():
    """Run the concurrency sweep test."""
    concurrency_levels = [1, 4, 16, 32, 64]

    async with httpx.AsyncClient() as client:
        for n in concurrency_levels:
            print(f"\n  Testing concurrency={n}...")
            try:
                results = await asyncio.gather(
                    *[send_request(client, i) for i in range(n)],
                    return_exceptions=True,
                )
                # Count exceptions and gibberish
                errors = sum(1 for r in results if isinstance(r, Exception))
                valid_results = [r for r in results if not isinstance(r, Exception)]
                gibberish_count = sum(1 for _, t in valid_results if is_gibberish(t))
                total_bad = errors + gibberish_count

                status = "PASS" if total_bad == 0 else "FAIL"
                print(f"    concurrency={n:>3}  gibberish={gibberish_count}/{n}  errors={errors}/{n}  [{status}]")

                if gibberish_count > 0:
                    for idx, t in sorted(valid_results)[:2]:
                        if is_gibberish(t):
                            print(f"      [{idx}] {t[:120]}")

                check(
                    f"concurrency={n} produces valid output",
                    total_bad == 0,
                    f"{gibberish_count} gibberish + {errors} errors out of {n} requests",
                )

            except Exception as e:
                check(f"concurrency={n} produces valid output", False, f"Exception: {e}")


def main():
    global checks_passed, checks_total

    print("=" * 60)
    print("Test Harness: sglang#21614 — Gibberish at high concurrency")
    print("=" * 60)

    # Kill any existing servers
    print("\nKilling existing sglang processes...")
    kill_server()

    # Start server
    print("\nStarting SGLang server...")
    server_proc, log_file = start_server()

    try:
        # Wait for server
        print("\nWaiting for server to be ready (may take 10-15 min for 397B model)...")
        if not wait_for_server(timeout=1200):
            check("Server starts successfully", False, "Server failed to start within timeout")
            score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
            print(f"\nResults: {checks_passed}/{checks_total} checks passed")
            print(f"SCORE: {score:.1f}")
            sys.exit(1)

        check("Server starts successfully", True)

        # Run concurrency tests
        print("\nRunning concurrency sweep test...")
        asyncio.run(run_concurrency_test())

    finally:
        # Cleanup: kill server process group
        print("\nCleaning up server process...")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        server_proc.kill()
        server_proc.wait()
        log_file.close()
        kill_server()

    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"\nResults: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
