#!/usr/bin/env python3
"""Test harness for vllm ROCm platform detection and CUDA initialization.

Tests (behavioral, subprocess-isolated):
  1. Import vllm.platforms in subprocess, verify torch.cuda is NOT initialized.
  2. After import, set CUDA_VISIBLE_DEVICES=0, verify device_count respects it.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"


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


def run_subprocess(script, timeout=120, env_override=None):
    env = os.environ.copy()
    for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
        env.pop(key, None)
    if env_override:
        env.update(env_override)
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/workspace",
        env=env,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-premature-cuda-init test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 1 (behavioral): importing vllm.platforms must NOT initialize CUDA.
# ---------------------------------------------------------------------------
print("\n--- Check 1: platform import does not initialize CUDA ---")

cuda_init_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import torch

pre_init = torch.cuda.is_initialized()
print(f"PRE_IMPORT_CUDA_INIT:{pre_init}")

if pre_init:
    print("SKIP:CUDA already initialized before import")
    sys.exit(0)

try:
    from vllm.platforms import current_platform
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

post_init = torch.cuda.is_initialized()
print(f"POST_IMPORT_CUDA_INIT:{post_init}")
print(f"PLATFORM:{current_platform}")
"""

try:
    stdout1, stderr1, rc1 = run_subprocess(cuda_init_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout1, rc1 = "TIMEOUT", -1

if "SKIP:CUDA already initialized" in stdout1:
    print("  [SKIP] CUDA was initialized before vllm import — test invalid")
elif "IMPORT:FAIL" in stdout1:
    err = stdout1.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import vllm.platforms", False, err)
elif "IMPORT:OK" in stdout1:
    check("Import vllm.platforms succeeds", True)

    check(
        "CUDA NOT initialized after vllm.platforms import",
        "POST_IMPORT_CUDA_INIT:False" in stdout1,
        "torch.cuda.is_initialized() is True — CUDA was prematurely initialized during import",
    )

# ---------------------------------------------------------------------------
# Check 2 (behavioral): after platform import, CUDA_VISIBLE_DEVICES should
# still be respected by torch.cuda.device_count().
# ---------------------------------------------------------------------------
print("\n--- Check 2: CUDA_VISIBLE_DEVICES respected after import ---")

device_count_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import torch

from vllm.platforms import current_platform

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count = torch.cuda.device_count()

if count == 0:
    print("NO_GPU")
else:
    print(f"DEVICE_COUNT:{count}")
    print(f"COUNT_IS_ONE:{count == 1}")
"""

try:
    stdout2, stderr2, rc2 = run_subprocess(device_count_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout2, rc2 = "TIMEOUT", -1

if "NO_GPU" in stdout2:
    print("  [SKIP] No GPU available — device count check skipped")
elif "DEVICE_COUNT:" in stdout2:
    check(
        "device_count=1 after setting CUDA_VISIBLE_DEVICES=0 post-import",
        "COUNT_IS_ONE:True" in stdout2,
        "device_count not 1 — CUDA may have been initialized before env var was set",
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
