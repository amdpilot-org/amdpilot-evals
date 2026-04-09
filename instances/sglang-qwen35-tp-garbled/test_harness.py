#!/usr/bin/env python3
"""Test harness for Qwen3.5-397B-A17B-FP8 garbage output with EAGLE spec decode.

Start server with EAGLE speculative decoding + aiter backend, send concurrent
requests, check for garbage/garbled output patterns.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time

_PY = "/opt/venv/bin/python3"
_PORT = 30000
_URL = f"http://localhost:{_PORT}"
_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8"
_TIMEOUT_STARTUP = 2400
_NUM_REQUESTS = 8

# Suspicious pattern regex from the issue
SUSP = re.compile(r'(ERER|spER|\b[A-Z]{8,}\b|[\\\"=]{5,}|!{20,})')

PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: Answer: $LETTER where LETTER is one of ABCD. Think step by step before answering.

Consider the Y-component of the intrinsic angular momentum operator for a spin-1/2 particle. Let A_y be represented by a 2x2 matrix satisfying A_y \u03c6 = a \u03c6. The matrix operator has the form A_y = c S, where c = h/4\u03c0 and S is the matrix [[0, -i], [i, 0]]. During the calculation, which statement below is correct?

A) The imaginary part of the eigenvalues is +2\u03c0 or -2\u03c0, and the real part is +h/4\u03c0 or -h/4\u03c0.
B) The eigenfunctions of A_y can also be eigenfunctions of A^2, but not of A_z.
C) The eigenfunctions of A_y are the basis functions of the matrix operator A_y given above.
D) The imaginary part of the eigenvalues is +1/2 or -1/2, and the real part is +1 or -1.
"""


def _find_snapshot(model_dir):
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_dir):
        return model_dir
    entries = os.listdir(snap_dir)
    if entries:
        return os.path.join(snap_dir, entries[0])
    return model_dir


def _kill_existing():
    subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-f", "sglang.srt"], capture_output=True)
    time.sleep(3)


def _wait_for_server():
    import urllib.request
    import urllib.error
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


async def _send_requests():
    """Send concurrent requests using httpx, matching the issue reproduction."""
    try:
        import httpx
    except ImportError:
        # Fallback to sequential urllib if httpx not available
        return _send_requests_urllib()

    results = []

    async def one(client, idx):
        payload = {
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "stream": False,
            "max_tokens": 512,
            "separate_reasoning": True,
        }
        try:
            r = await client.post(
                f"{_URL}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            j = r.json()
            m = j["choices"][0]["message"]
            rc = (m.get("reasoning_content") or "")[:400].replace("\n", " ")
            content = m.get("content") or ""
            suspicious = bool(SUSP.search(rc + content))
            return {
                "idx": idx,
                "status": r.status_code,
                "suspicious": suspicious,
                "reasoning_preview": rc[:100],
                "content_preview": content[:100],
            }
        except Exception as e:
            return {"idx": idx, "status": -1, "suspicious": True, "error": str(e)[:200]}

    async with httpx.AsyncClient() as client:
        tasks = [one(client, i) for i in range(_NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)

    return results


def _send_requests_urllib():
    """Fallback: send requests sequentially using urllib."""
    import urllib.request
    results = []
    for idx in range(_NUM_REQUESTS):
        payload = {
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "stream": False,
            "max_tokens": 512,
            "separate_reasoning": True,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            j = json.loads(resp.read())
            m = j["choices"][0]["message"]
            rc = (m.get("reasoning_content") or "")[:400].replace("\n", " ")
            content = m.get("content") or ""
            suspicious = bool(SUSP.search(rc + content))
            results.append({
                "idx": idx,
                "status": 200,
                "suspicious": suspicious,
                "reasoning_preview": rc[:100],
                "content_preview": content[:100],
            })
        except Exception as e:
            results.append({"idx": idx, "status": -1, "suspicious": True, "error": str(e)[:200]})
    return results


def main():
    print("=" * 60)
    print("Qwen3.5-397B-A17B-FP8 Garbage Output Test")
    print(f"  {_NUM_REQUESTS} concurrent requests with EAGLE spec decode")
    print("=" * 60)

    model_path = _find_snapshot(_MODEL)
    if not os.path.isdir(model_path):
        print(f"[FAIL] Model not found: {_MODEL}")
        print("SCORE: 0.0")
        return

    _kill_existing()

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"

    server_cmd = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", "4",
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--enable-aiter-allreduce-fusion",
        "--attention-backend", "triton",
        "--disable-radix-cache",
        "--mem-fraction-static", "0.85",
        "--reasoning-parser", "qwen3",
        "--port", str(_PORT),
    ]

    log_path = "/tmp/sglang_qwen35_garbled_test.log"
    log_file = open(log_path, "w")
    server = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("Waiting for server (model loading may take 15-30 min)...")
        if not _wait_for_server():
            print("[FAIL] Server did not become ready")
            # Check for crash in logs
            try:
                with open(log_path) as f:
                    log_tail = f.read()[-2000:]
                if "Error" in log_tail or "error" in log_tail:
                    print(f"  Log tail: {log_tail[-500:]}")
            except Exception:
                pass
            print("SCORE: 0.0")
            return

        print("Server ready. Sending concurrent requests...\n")

        results = asyncio.run(_send_requests())

        good = 0
        for r in results:
            idx = r["idx"]
            suspicious = r.get("suspicious", True)
            status = r.get("status", -1)

            if status == -1:
                print(f"  [{idx+1}/{_NUM_REQUESTS}] ERROR: {r.get('error', 'unknown')[:100]}")
            elif suspicious:
                preview = r.get("reasoning_preview", "")[:80]
                print(f"  [{idx+1}/{_NUM_REQUESTS}] GARBAGE: {preview}")
            else:
                preview = r.get("content_preview", "")[:80]
                print(f"  [{idx+1}/{_NUM_REQUESTS}] OK: {preview}")
                good += 1

        score = good / _NUM_REQUESTS * 100.0
        print(f"\n--- Results ---")
        print(f"  {good}/{_NUM_REQUESTS} valid responses ({score:.1f}%)")
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
