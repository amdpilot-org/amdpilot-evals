#!/usr/bin/env python3
"""Test harness v2 for vllm#35925: Qwen3.5-35B-A3B corrupted responses with AITER enabled.

Improvements over v1:
- 120 prompts (mix of text-only and multimodal) instead of 15
- TP=1 to match original bug report
- max_model_len=8192 for more KV cache utilization
- Multimodal inputs exercise the encoder attention path
- Safe kill pattern (won't kill the agent process)
- Statistically robust: at 7.8% corruption rate, 120 prompts gives
  ~99.9% probability of seeing at least one corruption if bug exists.
"""
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
TP = 1  # Match original bug report (single GPU)
NUM_TEXT_PROMPTS = 60
NUM_MULTIMODAL_PROMPTS = 60
TOTAL_PROMPTS = NUM_TEXT_PROMPTS + NUM_MULTIMODAL_PROMPTS

# Text-only prompts — diverse and requiring substantive responses
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


def check(name: str, condition: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}" + (f" -- {detail}" if detail else ""))


def is_corrupted(text: str) -> bool:
    """Detect corrupted output: repetitive single-character sequences.

    The AITER NaN bug produces outputs like "!!!!!!!!!!" or "?????????"
    — long runs of a single repeated character.
    """
    if not text or len(text.strip()) == 0:
        return False  # Empty output is not "corrupted" in the AITER sense
    # Check for 10+ consecutive identical non-whitespace characters
    if re.search(r'([^\s])\1{9,}', text):
        return True
    # Check if >60% of non-space content is a single punctuation character
    clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(clean) > 10:
        from collections import Counter
        counts = Counter(clean)
        most_char, most_count = counts.most_common(1)[0]
        if most_count / len(clean) > 0.6 and most_char in "!?.*#@&^~|/\\:;><":
            return True
    return False


def kill_gpu_processes():
    """Kill any leftover GPU processes safely (won't kill agent)."""
    os.system("pgrep -f 'python3 -m (sglang|vllm)' | xargs -r kill -9 2>/dev/null")
    time.sleep(5)


def run_generation_text_only(env_overrides: dict, prompts: list) -> list | None:
    """Run vLLM text-only generation and return outputs."""
    env = os.environ.copy()
    env.update(env_overrides)

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
            env=env, capture_output=True, text=True, timeout=1800,
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
        print(f"  stdout (last 500 chars): {result.stdout[-500:]}")
        return None

    import json
    outputs = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_TEXT:"):
            try:
                text = json.loads(line[len("OUTPUT_TEXT:"):])
                outputs.append(text)
            except json.JSONDecodeError:
                outputs.append(line[len("OUTPUT_TEXT:"):])
    return outputs


def run_generation_multimodal(env_overrides: dict, num_prompts: int) -> list | None:
    """Run vLLM multimodal generation with programmatically generated images."""
    env = os.environ.copy()
    env.update(env_overrides)

    # Generate multimodal prompts inside the subprocess to avoid
    # PIL dependency in the harness runner.
    script = f'''
import json
import io
import base64
import random

# Generate simple test images programmatically
def make_test_image(width=224, height=224, seed=0):
    """Create a simple colored image as bytes."""
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
        # Fallback: create a minimal valid PNG manually
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
        png_bytes = create_png(width, height,
                               random.randint(0,255), random.randint(0,255), random.randint(0,255))
        return io.BytesIO(png_bytes)

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
from vllm.multimodal.inputs import ImagePixelData

llm = LLM(
    model="{MODEL}",
    max_model_len={MAX_MODEL_LEN},
    trust_remote_code=True,
    tensor_parallel_size={TP},
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)

params = SamplingParams(temperature=0.0, max_tokens=200)

# Build multimodal prompts
num_prompts = {num_prompts}
prompts_data = []
for i in range(num_prompts):
    img_buf = make_test_image(seed=i)
    question = MULTIMODAL_QUESTIONS[i % len(MULTIMODAL_QUESTIONS)]
    prompt = f"<|im_start|>user\\n<image>\\n{{question}}<|im_end|>\\n<|im_start|>assistant\\n"
    try:
        from PIL import Image
        img = Image.open(img_buf)
    except ImportError:
        img = img_buf  # fallback
    prompts_data.append({{
        "prompt": prompt,
        "multi_modal_data": {{"image": img}},
    }})

outputs = llm.generate(prompts_data, params)
for o in outputs:
    text = o.outputs[0].text
    print("OUTPUT_TEXT:" + json.dumps(text))
print("GENERATION_DONE")
'''
    try:
        result = subprocess.run(
            [PYTHON, "-c", script],
            env=env, capture_output=True, text=True, timeout=2400,
        )
    except subprocess.TimeoutExpired:
        print("  Multimodal generation timed out after 2400s")
        return None

    if result.returncode != 0:
        print(f"  Multimodal generation failed (rc={result.returncode})")
        print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
        # If multimodal fails (e.g. no PIL, model doesn't support VLM),
        # fall back to text-only with extra prompts
        print("  Falling back to text-only generation with extra prompts")
        return None

    if "GENERATION_DONE" not in result.stdout:
        print("  Multimodal generation did not complete")
        return None

    import json
    outputs = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_TEXT:"):
            try:
                text = json.loads(line[len("OUTPUT_TEXT:"):])
                outputs.append(text)
            except json.JSONDecodeError:
                outputs.append(line[len("OUTPUT_TEXT:"):])
    return outputs


# ============================================================
# Test 1: Baseline -- AITER disabled should produce clean output
# ============================================================
print("\n=== Test 1: Baseline (AITER disabled, text-only) ===")
kill_gpu_processes()

baseline_env = {
    "VLLM_ROCM_USE_AITER": "0",
}

baseline_outputs = run_generation_text_only(baseline_env, TEXT_PROMPTS[:30])
check("Baseline generation completed", baseline_outputs is not None, "Generation failed")

if baseline_outputs is not None:
    baseline_corrupted = sum(1 for t in baseline_outputs if is_corrupted(t))
    check(
        "Baseline has no corrupted outputs",
        baseline_corrupted == 0,
        f"{baseline_corrupted}/{len(baseline_outputs)} corrupted",
    )
    for i, text in enumerate(baseline_outputs):
        if is_corrupted(text):
            print(f"    Baseline [{i}] CORRUPTED: {repr(text[:100])}")

# ============================================================
# Test 2: AITER enabled, text-only prompts (60 prompts)
# ============================================================
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
    check(
        "AITER text has no corrupted outputs",
        text_corrupted == 0,
        f"{text_corrupted}/{len(aiter_text_outputs)} corrupted",
    )
    for i, text in enumerate(aiter_text_outputs):
        if is_corrupted(text):
            print(f"    AITER text [{i}] CORRUPTED: {repr(text[:100])}")

# ============================================================
# Test 3: AITER enabled, multimodal prompts (60 prompts)
# ============================================================
print("\n=== Test 3: AITER enabled, multimodal (60 prompts) ===")
kill_gpu_processes()

aiter_mm_outputs = run_generation_multimodal(aiter_env, NUM_MULTIMODAL_PROMPTS)

if aiter_mm_outputs is not None:
    check("AITER multimodal generation completed", True)
    mm_corrupted = sum(1 for t in aiter_mm_outputs if is_corrupted(t))
    check(
        "AITER multimodal has no corrupted outputs",
        mm_corrupted == 0,
        f"{mm_corrupted}/{len(aiter_mm_outputs)} corrupted",
    )
    for i, text in enumerate(aiter_mm_outputs):
        if is_corrupted(text):
            print(f"    AITER mm [{i}] CORRUPTED: {repr(text[:100])}")
else:
    # Multimodal failed (no PIL / model doesn't support VLM) --
    # fall back to running more text prompts instead
    print("  Multimodal not available, running additional text prompts as fallback")
    extra_prompts = [f"Question {i}: Explain what the number {i*7} means in mathematics."
                     for i in range(60)]
    kill_gpu_processes()
    fallback_outputs = run_generation_text_only(aiter_env, extra_prompts)
    if fallback_outputs is not None:
        check("AITER fallback text generation completed", True)
        fb_corrupted = sum(1 for t in fallback_outputs if is_corrupted(t))
        check(
            "AITER fallback text has no corrupted outputs",
            fb_corrupted == 0,
            f"{fb_corrupted}/{len(fallback_outputs)} corrupted",
        )
        text_corrupted += fb_corrupted
    else:
        check("AITER fallback text generation completed", False, "Generation failed")

# ============================================================
# Test 4: Summary -- overall corruption rate
# ============================================================
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
    check(
        "Zero corruption across all AITER tests",
        total_corrupted == 0,
        f"{total_corrupted}/{total_aiter} corrupted ({corruption_rate:.1f}%)",
    )

# ============================================================
# Final Score
# ============================================================
kill_gpu_processes()

score = int(100 * checks_passed / checks_total) if checks_total > 0 else 0
print(f"\nSCORE: {score}")
sys.exit(0 if score == 100 else 1)
