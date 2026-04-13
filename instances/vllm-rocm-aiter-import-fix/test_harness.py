#!/usr/bin/env python3
"""Behavioral test: AITER backend import works with both env var states.

When VLLM_ROCM_USE_AITER=0, the backend must fall back to flash_attn
without crashing. When VLLM_ROCM_USE_AITER=1, AITER ops must load
correctly. Pre-fix code crashed with AttributeError when AITER was
disabled but an AITER backend was explicitly selected.
"""
import subprocess
import sys
import os
import textwrap

NUM_CHECKS = 4
results = {}


def run_subprocess(test_code: str, env_overrides: dict = None) -> tuple:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True, timeout=120, env=env
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# CHECK 1: Import succeeds on ROCm
check1_code = textwrap.dedent("""
import torch
if not torch.cuda.is_available() or 'gfx9' not in torch.cuda.get_device_properties(0).gcnArchName:
    print("IMPORT_SKIP")
    exit(1)
try:
    from vllm._aiter_ops import rocm_aiter_ops
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {e}")
    exit(1)
""")
ok, out = run_subprocess(check1_code)
if "IMPORT_SKIP" in out:
    print("SCORE: 0 (IMPORT_SKIP — not ROCm gfx9, auto-FAIL)")
    sys.exit(0)
results[1] = ok and "IMPORT_OK" in out
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — rocm_aiter_ops import")

# CHECK 2: VLLM_ROCM_USE_AITER=1 — aiter ops load correctly
check2_code = textwrap.dedent("""
import os
os.environ["VLLM_ROCM_USE_AITER"] = "1"
from vllm._aiter_ops import rocm_aiter_ops
rocm_aiter_ops.reload_envs()

if rocm_aiter_ops.is_enabled():
    # Verify key ops are accessible
    try:
        _ = rocm_aiter_ops.flash_attn_varlen_func
        _ = rocm_aiter_ops.pa_fwd_asm
        print("AITER_ON_OK")
    except AttributeError as e:
        print(f"AITER_ON_FAIL: {e}")
else:
    # aiter package not installed — acceptable
    print("AITER_NOT_INSTALLED")
""")
ok, out = run_subprocess(check2_code, {"VLLM_ROCM_USE_AITER": "1"})
results[2] = ok and ("AITER_ON_OK" in out or "AITER_NOT_INSTALLED" in out)
print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — VLLM_ROCM_USE_AITER=1 ops accessible")

# CHECK 3: VLLM_ROCM_USE_AITER=0 — no crash, falls back gracefully
check3_code = textwrap.dedent("""
import os
os.environ["VLLM_ROCM_USE_AITER"] = "0"
from vllm._aiter_ops import rocm_aiter_ops
rocm_aiter_ops.reload_envs()

if rocm_aiter_ops.is_enabled():
    print("AITER_OFF_FAIL: should be disabled with VLLM_ROCM_USE_AITER=0")
else:
    # Verify accessing ops when disabled doesn't crash
    try:
        fa = rocm_aiter_ops.flash_attn_varlen_func
        pa = rocm_aiter_ops.pa_fwd_asm
        # These should return the flash_attn fallback, not crash
        print("AITER_OFF_OK")
    except AttributeError as e:
        print(f"AITER_OFF_CRASH: {e}")
""")
ok, out = run_subprocess(check3_code, {"VLLM_ROCM_USE_AITER": "0"})
results[3] = ok and "AITER_OFF_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — VLLM_ROCM_USE_AITER=0 no crash, graceful fallback")

# CHECK 4: Toggle env var — state changes correctly
check4_code = textwrap.dedent("""
import os
from vllm._aiter_ops import rocm_aiter_ops

os.environ["VLLM_ROCM_USE_AITER"] = "1"
rocm_aiter_ops.reload_envs()
state_on = rocm_aiter_ops.is_enabled()

os.environ["VLLM_ROCM_USE_AITER"] = "0"
rocm_aiter_ops.reload_envs()
state_off = rocm_aiter_ops.is_enabled()

if state_off == False:
    # state_on may be True or False depending on whether aiter is installed
    print("TOGGLE_OK")
else:
    print(f"TOGGLE_FAIL: after setting =0, is_enabled still returns {state_off}")
""")
ok, out = run_subprocess(check4_code)
results[4] = ok and "TOGGLE_OK" in out
print(f"CHECK 4: {'PASS' if results[4] else 'FAIL'} — env var toggle changes enabled state")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
