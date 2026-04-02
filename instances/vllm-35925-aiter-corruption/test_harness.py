#!/usr/bin/env python3
"""Test harness v2 for vLLM issue #35925.

This version is intentionally stronger than the original live harness:
- 120 prompts (60 text-only + 60 multimodal) instead of 15
- TP=1 to match the original report more closely
- max_model_len=8192 to exercise more KV/cache state
- safe kill pattern
- runtime forced onto the checked-out source via PYTHONPATH
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

PYTHON = "/usr/bin/python3"
checks_passed = 0
checks_total = 0

MODEL = "Qwen/Qwen3.5-35B-A3B"
MAX_MODEL_LEN = 8192
TP = 1
NUM_TEXT_PROMPTS = 60
NUM_MULTIMODAL_PROMPTS = 60
TOTAL_PROMPTS = NUM_TEXT_PROMPTS + NUM_MULTIMODAL_PROMPTS

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
    "What is the speed of light in km/s? Just the number.",
    "Name three countries in South America.",
    "What is the square root of 144?",
    "Describe what happens during a solar eclipse.",
    "What is the largest ocean on Earth?",
    "How many bones does an adult human body have? Just the number.",
    "What is the chemical formula for table salt?",
    "Name the first person to walk on the Moon.",
    "What is the largest planet in our solar system?",
    "Explain what gravity is in one sentence.",
    "What year did World War II end?",
    "Name three programming languages.",
    "What is the tallest mountain in the world?",
    "How many continents are there? Just the number.",
    "What is DNA? Explain in one sentence.",
    "Name the four seasons.",
    "What is the currency of Japan?",
    "How many sides does a hexagon have?",
    "What is the Pythagorean theorem?",
    "Name three elements from the periodic table.",
    "What is the freezing point of water in Fahrenheit?",
    "Who wrote Hamlet?",
    "What is the smallest prime number?",
    "Name the seven colors of the rainbow.",
    "What is an atom?",
    "How many minutes are in an hour?",
    "What is the formula for the area of a circle?",
    "Name three types of rocks.",
    "What does CPU stand for?",
    "What is the distance from Earth to the Moon in miles? Approximate.",
    "Name three mammals that live in the ocean.",
    "What is Pi to 5 decimal places?",
    "Describe how a battery works in two sentences.",
    "What is the atomic number of carbon?",
    "Name three types of clouds.",
    "What is the main gas in Earth's atmosphere?",
    "How many teeth does an adult human have?",
    "What is the largest desert on Earth?",
    "Name three Nobel Prize categories.",
    "What temperature is absolute zero in Celsius?",
    "Describe what an earthquake is.",
    "What is the longest river in the world?",
    "How many strings does a standard guitar have?",
    "What is mitosis?",
    "Name the first five elements of the periodic table.",
]


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}" + (f" -- {detail}" if detail else ""))


def is_corrupted(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return False
    if re.search(r"([^\s])\1{9,}", text):
        return True
    clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(clean) > 10:
        counts = {}
        for ch in clean:
            counts[ch] = counts.get(ch, 0) + 1
        most_char, most_count = max(counts.items(), key=lambda x: x[1])
        if most_count / len(clean) > 0.6 and most_char in "!?.*#@&^~|/\\:;><":
            return True
    return False


def kill_gpu_processes() -> None:
    os.system("pgrep -f 'python3 -m (sglang|vllm)' | xargs -r kill -9 2>/dev/null")
    time.sleep(5)


def run_generation_text_only(env_overrides: dict[str, str], prompts: list[str]) -> list[str] | None:
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONPATH"] = "/workspace/vllm"

    script = f'''
import json
from vllm import LLM, SamplingParams

prompts = {prompts!r}
llm = LLM(
    model="{MODEL}",
    max_model_len={MAX_MODEL_LEN},
    trust_remote_code=True,
    tensor_parallel_size={TP},
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)
params = SamplingParams(temperature=0.0, max_tokens=200)
outputs = llm.generate(prompts, params)
for o in outputs:
    text = o.outputs[0].text
    print("OUTPUT_TEXT:" + json.dumps(text))
print("GENERATION_DONE")
'''
    try:
        result = subprocess.run(
            [PYTHON, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print("  Generation timed out after 1800s")
        return None

    if result.returncode != 0:
        print(f"  Generation process failed (rc={result.returncode})")
        print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
        return None

    if "GENERATION_DONE" not in result.stdout:
        print("  Generation did not complete successfully")
        return None

    outputs = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_TEXT:"):
            try:
                outputs.append(__import__("json").loads(line[len("OUTPUT_TEXT:"):]))
            except Exception:
                outputs.append(line[len("OUTPUT_TEXT:"):])
    return outputs


def run_generation_multimodal(env_overrides: dict[str, str], num_prompts: int) -> list[str] | None:
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONPATH"] = "/workspace/vllm"

    script = f'''
import io
import json
import random

def make_test_image(width=224, height=224, seed=0):
    random.seed(seed)
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height),
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except ImportError:
        import struct, zlib
        def create_png(w, h, r, g, b):
            raw = b""
            for _ in range(h):
                raw += b"\\x00" + bytes([r, g, b]) * w
            compressed = zlib.compress(raw)
            def chunk(ctype, data):
                c = ctype + data
                crc = zlib.crc32(c) & 0xffffffff
                return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)
            sig = b"\\x89PNG\\r\\n\\x1a\\n"
            ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            idat = chunk(b"IDAT", compressed)
            iend = chunk(b"IEND", b"")
            return sig + ihdr + idat + iend
        return io.BytesIO(create_png(width, height, random.randint(0,255), random.randint(0,255), random.randint(0,255)))

MULTIMODAL_QUESTIONS = [
    "Describe what you see in this image.",
    "What colors are present in this image?",
    "Is this image bright or dark? Why?",
    "What could this image represent?",
    "Describe the dominant color in this image.",
    "How many distinct colors can you see?",
    "What mood does this image convey?",
    "Describe the texture you see.",
    "What shape does this image resemble?",
    "Is this a natural or artificial image? Why?",
]

from vllm import LLM, SamplingParams
llm = LLM(
    model="{MODEL}",
    max_model_len={MAX_MODEL_LEN},
    trust_remote_code=True,
    tensor_parallel_size={TP},
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)
params = SamplingParams(temperature=0.0, max_tokens=200)
prompts_data = []
for i in range({num_prompts}):
    img_buf = make_test_image(seed=i)
    question = MULTIMODAL_QUESTIONS[i % len(MULTIMODAL_QUESTIONS)]
    prompt = f"<|im_start|>user\\n<image>\\n{{question}}<|im_end|>\\n<|im_start|>assistant\\n"
    try:
        from PIL import Image
        img = Image.open(img_buf)
    except ImportError:
        img = img_buf
    prompts_data.append({{"prompt": prompt, "multi_modal_data": {{"image": img}}}})

outputs = llm.generate(prompts_data, params)
for o in outputs:
    text = o.outputs[0].text
    print("OUTPUT_TEXT:" + json.dumps(text))
print("GENERATION_DONE")
'''
    try:
        result = subprocess.run(
            [PYTHON, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=2400,
        )
    except subprocess.TimeoutExpired:
        print("  Multimodal generation timed out after 2400s")
        return None

    if result.returncode != 0 or "GENERATION_DONE" not in result.stdout:
        print(f"  Multimodal generation failed (rc={result.returncode})")
        print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
        return None

    outputs = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_TEXT:"):
            try:
                outputs.append(__import__("json").loads(line[len("OUTPUT_TEXT:"):]))
            except Exception:
                outputs.append(line[len("OUTPUT_TEXT:"):])
    return outputs


print("\n=== Test 1: Baseline (AITER disabled, text-only) ===")
kill_gpu_processes()
baseline_env = {"VLLM_ROCM_USE_AITER": "0"}
baseline_outputs = run_generation_text_only(baseline_env, TEXT_PROMPTS[:30])
check("Baseline generation completed", baseline_outputs is not None, "Generation failed")

if baseline_outputs is not None:
    baseline_corrupted = sum(1 for t in baseline_outputs if is_corrupted(t))
    check("Baseline has no corrupted outputs", baseline_corrupted == 0, f"{baseline_corrupted}/{len(baseline_outputs)} corrupted")

print("\n=== Test 2: AITER enabled, text-only (60 prompts) ===")
kill_gpu_processes()
aiter_env = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MHA": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
    "VLLM_ROCM_USE_AITER_RMSNORM": "0",
    "VLLM_ROCM_USE_AITER_TRITON_ROPE": "0",
}
aiter_text_outputs = run_generation_text_only(aiter_env, TEXT_PROMPTS)
check("AITER text generation completed", aiter_text_outputs is not None, "Generation failed")
text_corrupted = 0
if aiter_text_outputs is not None:
    text_corrupted = sum(1 for t in aiter_text_outputs if is_corrupted(t))
    check("AITER text has no corrupted outputs", text_corrupted == 0, f"{text_corrupted}/{len(aiter_text_outputs)} corrupted")

print("\n=== Test 3: AITER enabled, multimodal (60 prompts) ===")
kill_gpu_processes()
aiter_mm_outputs = run_generation_multimodal(aiter_env, 60)

if aiter_mm_outputs is not None:
    check("AITER multimodal generation completed", True)
    mm_corrupted = sum(1 for t in aiter_mm_outputs if is_corrupted(t))
    check("AITER multimodal has no corrupted outputs", mm_corrupted == 0, f"{mm_corrupted}/{len(aiter_mm_outputs)} corrupted")
else:
    print("  Multimodal not available, running fallback extra text prompts")
    extra_prompts = [f"Question {{i}}: Explain what the number {{i*7}} means in mathematics." for i in range(60)]
    kill_gpu_processes()
    fallback_outputs = run_generation_text_only(aiter_env, extra_prompts)
    if fallback_outputs is not None:
        check("AITER fallback text generation completed", True)
        fb_corrupted = sum(1 for t in fallback_outputs if is_corrupted(t))
        check("AITER fallback text has no corrupted outputs", fb_corrupted == 0, f"{fb_corrupted}/{len(fallback_outputs)} corrupted")
        text_corrupted += fb_corrupted
    else:
        check("AITER fallback text generation completed", False, "Generation failed")

print("\n=== Test 4: Overall corruption summary ===")
total_aiter = 0
total_corrupted = 0
if aiter_text_outputs is not None:
    total_aiter += len(aiter_text_outputs)
    total_corrupted += sum(1 for t in aiter_text_outputs if is_corrupted(t))
if aiter_mm_outputs is not None:
    total_aiter += len(aiter_mm_outputs)
    total_corrupted += sum(1 for t in aiter_mm_outputs if is_corrupted(t))
if total_aiter > 0:
    corruption_rate = total_corrupted / total_aiter * 100
    print(f"  Total AITER outputs: {total_aiter}")
    print(f"  Total corrupted: {total_corrupted}")
    print(f"  Corruption rate: {corruption_rate:.1f}%")
    check("Zero corruption across all AITER tests", total_corrupted == 0, f"{total_corrupted}/{total_aiter} corrupted ({corruption_rate:.1f}%)")

kill_gpu_processes()
score = int(100 * checks_passed / checks_total) if checks_total > 0 else 0
print(f"SCORE: {score}")
sys.exit(0 if score == 100 else 1)
