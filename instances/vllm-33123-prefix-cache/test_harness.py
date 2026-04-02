#!/usr/bin/env python3
"""Eval bundle harness for vLLM issue #33123."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

PYTHON = "/usr/bin/python3"
MODEL_NAME = "Qwen/Qwen3-0.6B"
HOST = "127.0.0.1"
PORT = 8192
NUM_REQUESTS = 20
MAX_NEW_TOKENS = 10

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def kill_gpu_processes() -> None:
    os.system("pgrep -f 'python3 -m (sglang|vllm)' | xargs -r kill -9")
    time.sleep(3)


def wait_for_server(timeout: int = 600) -> bool:
    url = f"http://{HOST}:{PORT}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(5)
    return False


def get_prompt_token_ids():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return token_ids


def send_completion_request(prompt_token_ids):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_token_ids,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "logprobs": 5,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    kill_gpu_processes()

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = "/workspace/vllm"
    server_cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--dtype", "bfloat16",
        "--max-model-len", "1024",
        "--enforce-eager",
        "--host", HOST,
        "--port", str(PORT),
        "--generation-config", "vllm",
        "--gpu-memory-utilization", "0.85",
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        check("server_startup", wait_for_server(), "server did not become ready")
        if checks_passed != checks_total:
            score = int(100 * checks_passed / checks_total)
            print(f"SCORE: {score}")
            return 1

        prompt_token_ids = get_prompt_token_ids()
        outputs = []
        for i in range(NUM_REQUESTS):
            try:
                resp = send_completion_request(prompt_token_ids)
                text = resp["choices"][0]["text"]
                outputs.append(text)
                print(f"  Request {i+1:02d}: {text!r}")
            except Exception as e:  # noqa: BLE001
                outputs.append(None)
                print(f"  Request {i+1:02d}: ERROR — {e}")

        valid = [o for o in outputs if o is not None]
        check("all_requests_valid", len(valid) == NUM_REQUESTS, f"{len(valid)}/{NUM_REQUESTS}")
        if valid:
            first = valid[0]
            check("all_requests_identical", all(o == first for o in valid), "cache miss/hit diverged")
            if len(valid) >= 5:
                hit_window = valid[1:]
                check("cache_hit_requests_consistent", all(o == hit_window[0] for o in hit_window), "hit path unstable")
        else:
            check("all_requests_identical", False, "no valid outputs")
            check("cache_hit_requests_consistent", False, "no valid outputs")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        kill_gpu_processes()

    score = int(100 * checks_passed / checks_total) if checks_total else 0
    print(f"SCORE: {score}")
    return 0 if score == 100 else 1


if __name__ == "__main__":
    sys.exit(main())
