#!/usr/bin/env python3
"""Eval bundle harness for vLLM issue #35925.

The bundle is intentionally pinned to a pre-fix commit. The harness mixes
text-only and multimodal prompts so the bug is not silently missed by a weak
text-only check.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

PYTHON = "/usr/bin/python3"
MODEL = "Qwen/Qwen3.5-35B-A3B"
HOST = "127.0.0.1"
PORT = 8192
TP = 4
MAX_MODEL_LEN = 4096

TEXT_PROMPTS = [
    "What is 2+2? Give only the numerical answer.",
    "Name the capital of France in one word.",
    "What color is the sky on a clear day? One word answer.",
    "What is the chemical symbol for water?",
    "How many days are in a week? Just the number.",
    "What planet is closest to the Sun? One word.",
    "What is 10 times 5? Just the number.",
    "Name one primary color.",
    "What language is spoken in Brazil? One word.",
    "What is the boiling point of water in Celsius? Just the number.",
    "Explain briefly what photosynthesis is.",
    "Write a short poem about a cat.",
    "Describe the process of making tea step by step.",
    "What are the three states of matter? List them.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "List three mammals.",
    "What is the largest ocean on Earth?",
    "Give one adjective describing snow.",
    "What does CPU stand for?",
    "How many letters are in the English alphabet?",
]

MULTIMODAL_PROMPTS = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one short sentence."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5Wn6sAAAAASUVORK5CYII="}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is the image? Reply with one word."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5Wn6sAAAAASUVORK5CYII="}},
        ],
    },
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


def is_corrupted(text: str) -> bool:
    if re.search(r"(.)\1{9,}", text):
        return True
    if len(text) > 10:
        clean = text.replace(" ", "").replace("\n", "")
        if clean:
            counts = {}
            for ch in clean:
                counts[ch] = counts.get(ch, 0) + 1
            ch, cnt = max(counts.items(), key=lambda x: x[1])
            if cnt / len(clean) > 0.6 and ch in "!?.*#@&^~":
                return True
    return False


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
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    return False


def start_server(aiter_enabled: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = "/workspace/vllm"
    env["VLLM_ROCM_USE_AITER"] = "1" if aiter_enabled else "0"
    env["VLLM_ROCM_USE_AITER_MHA"] = "1" if aiter_enabled else "0"
    env["VLLM_ROCM_USE_AITER_MOE"] = "1" if aiter_enabled else "0"
    env["VLLM_ROCM_USE_AITER_LINEAR"] = "0"
    env["VLLM_ROCM_USE_AITER_RMSNORM"] = "0"
    env["VLLM_ROCM_USE_AITER_TRITON_ROPE"] = "0"
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--host", HOST,
        "--port", str(PORT),
        "--tensor-parallel-size", str(TP),
        "--gpu-memory-utilization", "0.95",
        "--generation-config", "vllm",
        "--enforce-eager",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def send_completion(prompt: str) -> str | None:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 200,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["text"]
    except Exception as e:  # noqa: BLE001
        print(f"    completion error: {e}")
        return None


def send_multimodal(messages: list[dict]) -> str | None:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        print(f"    multimodal error: {e}")
        return None


def run_suite(aiter_enabled: bool) -> list[str | None]:
    server = start_server(aiter_enabled)
    try:
        if not wait_for_server():
            return []
        outputs = [send_completion(prompt) for prompt in TEXT_PROMPTS]
        outputs.extend(send_multimodal([mm_prompt]) for mm_prompt in MULTIMODAL_PROMPTS)
        return outputs
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        kill_gpu_processes()


def main() -> int:
    kill_gpu_processes()

    print("=== baseline: AITER disabled ===")
    baseline_outputs = run_suite(False)
    check("baseline_completed", len(baseline_outputs) == len(TEXT_PROMPTS) + len(MULTIMODAL_PROMPTS))
    baseline_corrupt = [o for o in baseline_outputs if o and is_corrupted(o)]
    check("baseline_clean", len(baseline_corrupt) == 0, f"{len(baseline_corrupt)} corrupted")

    print("=== AITER enabled ===")
    aiter_outputs = run_suite(True)
    check("aiter_completed", len(aiter_outputs) == len(TEXT_PROMPTS) + len(MULTIMODAL_PROMPTS))
    aiter_corrupt = [o for o in aiter_outputs if o and is_corrupted(o)]
    check("aiter_clean", len(aiter_corrupt) == 0, f"{len(aiter_corrupt)} corrupted")

    if baseline_outputs and aiter_outputs and len(baseline_outputs) == len(aiter_outputs):
        paired = [(b, a) for b, a in zip(baseline_outputs, aiter_outputs) if b is not None and a is not None]
        check("all_outputs_nonempty", len(paired) == len(TEXT_PROMPTS) + len(MULTIMODAL_PROMPTS))
        check(
            "aiter_matches_baseline_shape",
            all(len(a.strip()) > 0 for _, a in paired),
            "some AITER outputs empty",
        )
    else:
        check("output_pairing", False, "missing outputs")

    score = int(100 * checks_passed / checks_total) if checks_total else 0
    print(f"SCORE: {score}")
    return 0 if score == 100 else 1


if __name__ == "__main__":
    sys.exit(main())
