#!/usr/bin/env python3
"""Test harness for sglang MLA decode NaN-on-padded-rows issue.

Tests:
  1. Verify the MLA decode function exists in the attention backend
  2. Check that the non-CUDA-graph MLA decode path correctly extracts
     valid output rows without selecting unwritten padded positions
  3. Start server and verify basic inference works (no crashes)

The bug: padded attention layouts create gaps between sequences in the
output buffer. If the extraction logic uses the wrong indexing scheme,
it reads from unwritten padded positions, producing NaN or garbage values.
"""

import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

_PY = "/opt/venv/bin/python3"
_PORT = 30000
_URL = f"http://localhost:{_PORT}"
_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b"
_TIMEOUT_STARTUP = 2400
_TIMEOUT_REQUEST = 300

_AITER_BACKEND = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"

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


def _kill_existing():
    subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(2)


def _wait_for_server():
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


def _send_request(prompt):
    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
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


def main():
    print("=" * 60)
    print("SGLang MLA Decode NaN-Padded Test")
    print("=" * 60)

    # --- Check 1: aiter_backend.py exists ---
    print("\n--- Check 1: File exists ---")
    if not check("aiter_backend.py exists", os.path.isfile(_AITER_BACKEND)):
        print(f"\nSCORE: 0.0")
        return

    # Read source
    with open(_AITER_BACKEND) as f:
        source = f.read()

    # --- Check 2: Find the MLA decode code block with pad_sequence_with_mask ---
    print("\n--- Check 2: MLA decode padded output extraction ---")

    # Find the code region that uses pad_sequence_with_mask + mla_decode_fwd
    # The bug: output extraction doesn't account for the layout mismatch
    # between the padded input layout and the kernel's contiguous output ordering

    # Find all blocks that contain pad_sequence_with_mask
    pad_mask_regions = []
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "pad_sequence_with_mask" in line and "def " not in line and "#" not in line.lstrip()[:1]:
            # Extract surrounding context (50 lines after)
            region = "\n".join(lines[i:min(i+50, len(lines))])
            pad_mask_regions.append((i + 1, region))

    found_pad_mask = len(pad_mask_regions) > 0
    check("pad_sequence_with_mask usage found in MLA decode",
          found_pad_mask,
          "pad_sequence_with_mask not found in attention backend")

    if not found_pad_mask:
        # If pad_sequence_with_mask isn't used at all, the fix may have
        # removed the padded path entirely — check that output extraction
        # accounts for kernel output ordering
        has_valid_extract = bool(re.search(
            r'o\s*\[.*\]|output.*slice|output.*extract',
            source
        ))
        check("MLA decode uses correct output extraction",
              has_valid_extract,
              "No output extraction logic found")
    else:
        # Check each region — the output extraction must account for the
        # layout mismatch between padded input and contiguous kernel output
        uses_mask_indexing = False
        uses_correct_extract = False

        for lineno, region in pad_mask_regions:
            # Check for the buggy pattern: indexing output with the input mask
            if re.search(r'return\s+o\s*\[\s*q_mask\s*\]', region):
                uses_mask_indexing = True
            # Check for any extraction that accounts for contiguous ordering
            if re.search(r'o\s*\[\s*:.*\]', region) and not re.search(r'o\s*\[\s*q_mask\s*\]', region):
                uses_correct_extract = True
            if re.search(r'qo_indptr.*item', region):
                uses_correct_extract = True

        if uses_mask_indexing and not uses_correct_extract:
            check("MLA decode output extraction handles padding correctly",
                  False,
                  "output extraction does not account for layout mismatch "
                  "between padded input and kernel output ordering")
        elif uses_correct_extract:
            check("MLA decode output extraction handles padding correctly",
                  True)
        else:
            has_safe = not uses_mask_indexing
            check("MLA decode output extraction handles padding correctly",
                  has_safe,
                  "could not determine extraction method")

    # --- Check 3: Server starts and inference works ---
    print("\n--- Check 3: Server inference sanity ---")

    _kill_existing()

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    server_cmd = [
        _PY, "-m", "sglang.launch_server",
        "--model-path", _MODEL,
        "--tp", "4",
        "--attention-backend", "aiter",
        "--disable-radix-cache",
        "--mem-fraction-static", "0.8",
        "--port", str(_PORT),
    ]

    log_file = open("/tmp/sglang_nan_test.log", "w")
    server = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("  Waiting for server (model loading ~15-30 min)...")
        if not _wait_for_server():
            check("Server starts and produces valid output", False,
                  "server did not become ready")
        else:
            # Send a few test requests
            test_prompts = [
                "What is 2 + 2?",
                "Name three colors.",
                "What is the capital of Japan?",
            ]
            good = 0
            for prompt in test_prompts:
                response = _send_request(prompt)
                if response and len(response.strip()) > 3:
                    good += 1

            check("Server starts and produces valid output",
                  good >= 2,
                  f"only {good}/{len(test_prompts)} valid responses")

    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log_file.close()
        _kill_existing()

    # --- Summary ---
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
