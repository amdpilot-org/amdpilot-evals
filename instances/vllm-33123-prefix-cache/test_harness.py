#!/usr/bin/env python3
"""Improved harness for vLLM issue #33123."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

PYTHON = "/usr/bin/python3"
MODEL_NAME = "Qwen/Qwen3-0.6B"
HOST = "127.0.0.1"
PORT_PC_ON = 8192
PORT_PC_OFF = 8193
NUM_REQUESTS = 6
MAX_NEW_TOKENS = 30
LOGPROB_TOLERANCE = 1e-3

PROMPT_MESSAGES = [
    [{"role": "user", "content": "What is 2+2?"}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ],
    [
        {"role": "system", "content": "You are a knowledgeable geography expert."},
        {"role": "user", "content": "List the founding members of the European Union and explain the significance of the Treaty of Rome in establishing economic cooperation."},
    ],
]

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


def kill_vllm_servers() -> None:
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            pids = [p for p in pids if int(p) != my_pid]
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass
    time.sleep(2)


def wait_for_server(host: str, port: int, timeout: int = 600) -> bool:
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  Server on :{port} ready after {time.time() - start:.1f}s")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(5)
    print(f"  Server on :{port} did not become ready within {timeout}s")
    return False


def get_prompt_token_ids_list():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = []
    labels = ["short", "medium", "long"]
    for i, messages in enumerate(PROMPT_MESSAGES):
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        result.append((token_ids, labels[i], messages))
        print(f"  Prompt '{labels[i]}': {len(token_ids)} tokens")
    return result


def extract_logprobs(response):
    try:
        lp = response["choices"][0]["logprobs"]
        if lp and "token_logprobs" in lp:
            return lp["token_logprobs"]
    except (KeyError, IndexError, TypeError):
        pass
    return None


def compare_logprobs(lp1, lp2, tolerance: float = LOGPROB_TOLERANCE):
    if lp1 is None or lp2 is None:
        return True, 0.0
    min_len = min(len(lp1), len(lp2))
    if min_len == 0:
        return True, 0.0
    max_diff = 0.0
    for a, b in zip(lp1[:min_len], lp2[:min_len]):
        if a is None or b is None:
            continue
        max_diff = max(max_diff, abs(a - b))
    return max_diff < tolerance, max_diff


def start_vllm_server(port: int, enable_prefix_caching: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/vllm"
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--dtype", "bfloat16",
        "--max-model-len", "1024",
        "--enforce-eager",
        "--host", HOST,
        "--port", str(port),
        "--gpu-memory-utilization", "0.85",
    ]
    if not enable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def send_completion_request(host: str, port: int, prompt_token_ids):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_token_ids,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "logprobs": 1,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=f"http://{host}:{port}/v1/completions", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_determinism_for_prompt(host: str, port: int, prompt_token_ids, label: str):
    texts = []
    logprobs_list = []
    for i in range(NUM_REQUESTS):
        try:
            resp = send_completion_request(host, port, prompt_token_ids)
            text = resp["choices"][0]["text"]
            lp = extract_logprobs(resp)
            texts.append(text)
            logprobs_list.append(lp)
            print(f"    Request {i+1}/{NUM_REQUESTS}: {text[:60]!r}")
        except Exception as e:
            texts.append(None)
            logprobs_list.append(None)
            print(f"    Request {i+1}/{NUM_REQUESTS}: ERROR — {e}")
    return texts, logprobs_list


def main() -> int:
    kill_vllm_servers()
    print("Tokenizing prompts...")
    prompts = get_prompt_token_ids_list()

    print("\n" + "=" * 60)
    print("PHASE 1: Prefix caching ENABLED (testing for bug)")
    print("=" * 60)
    server_proc = start_vllm_server(PORT_PC_ON, enable_prefix_caching=True)
    try:
        if not wait_for_server(HOST, PORT_PC_ON, timeout=600):
            check("server_startup_pc_on", False, "Server did not start")
            for _ in range(9):
                check("skipped", False, "Server not available")
            score = int(100 * checks_passed / checks_total) if checks_total > 0 else 0
            print(f"SCORE: {score}")
            return score

        check("server_startup_pc_on", True)

        all_prompts_text_match = True
        all_prompts_logprob_match = True
        any_first_vs_second_mismatch = False

        for token_ids, label, _messages in prompts:
            print(f"\n  Testing prompt '{label}' ({len(token_ids)} tokens)...")
            texts, logprobs_list = test_determinism_for_prompt(HOST, PORT_PC_ON, token_ids, label)
            valid = [t for t in texts if t is not None]
            if len(valid) < 2:
                all_prompts_text_match = False
                continue

            if valid[0] != valid[1]:
                print(f"    DIVERGENCE on '{label}': run1={valid[0][:40]!r} vs run2={valid[1][:40]!r}")
                any_first_vs_second_mismatch = True
                all_prompts_text_match = False

            if not all(t == valid[0] for t in valid):
                all_prompts_text_match = False

            valid_lp = [lp for lp in logprobs_list if lp is not None]
            if len(valid_lp) >= 2:
                for i in range(1, len(valid_lp)):
                    is_close, max_diff = compare_logprobs(valid_lp[0], valid_lp[i])
                    if not is_close:
                        print(f"    LOGPROB DIVERGENCE on '{label}': run1 vs run{i+1}, max_diff={max_diff:.6f}")
                        all_prompts_logprob_match = False

        check("pc_on_first_matches_second_all_prompts", not any_first_vs_second_mismatch, "First request differs from second request")
        check("pc_on_all_requests_identical_text", all_prompts_text_match, "Output text varies across requests")
        check("pc_on_logprobs_consistent", all_prompts_logprob_match, f"Logprob differences exceed tolerance ({LOGPROB_TOLERANCE})")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        kill_vllm_servers()

    print("\n" + "=" * 60)
    print("PHASE 2: Prefix caching DISABLED (negative control)")
    print("=" * 60)
    server_proc = start_vllm_server(PORT_PC_OFF, enable_prefix_caching=False)
    try:
        if not wait_for_server(HOST, PORT_PC_OFF, timeout=600):
            check("server_startup_pc_off", False, "Negative control server did not start")
        else:
            check("server_startup_pc_off", True)
            all_stable = True
            for token_ids, label, _messages in prompts:
                texts, _logprobs = test_determinism_for_prompt(HOST, PORT_PC_OFF, token_ids, label)
                valid = [t for t in texts if t is not None]
                if len(valid) < 2 or not all(t == valid[0] for t in valid):
                    all_stable = False
            check("pc_off_negative_control_stable", all_stable, "Negative control itself is unstable")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
        kill_vllm_servers()

    score = int(100 * checks_passed / checks_total) if checks_total else 0
    print(f"SCORE: {score}")
    return 0 if score == 100 else 1


if __name__ == "__main__":
    sys.exit(main())
