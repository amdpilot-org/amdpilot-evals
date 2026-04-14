#!/usr/bin/env python3
"""Behavioral test: paged_mqa_logits_module caching behavior.

The helper must return the same module object on repeated calls (lru_cache),
not re-import the module each time. Pre-fix code reloaded the Triton/aiter
module on every call, adding per-call import overhead.
"""
import subprocess
import sys
import os
import textwrap

NUM_CHECKS = 3
results = {}


def run_subprocess(test_code: str) -> tuple:
    proc = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True, timeout=120, env=os.environ.copy()
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# CHECK 1: Import succeeds
check1_code = textwrap.dedent("""
import torch
if not torch.cuda.is_available() or 'gfx9' not in torch.cuda.get_device_properties(0).gcnArchName:
    print("IMPORT_SKIP")
    exit(1)
try:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import paged_mqa_logits_module
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
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — paged_mqa_logits_module import")

# CHECK 2: Repeated calls return same object (identity check)
check2_code = textwrap.dedent("""
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import paged_mqa_logits_module

mod1 = paged_mqa_logits_module()
mod2 = paged_mqa_logits_module()
mod3 = paged_mqa_logits_module()

if mod1 is None:
    print("MODULE_NONE")  # aiter not installed — acceptable skip
elif mod1 is mod2 and mod2 is mod3:
    print("IDENTITY_OK")
else:
    print(f"IDENTITY_FAIL: id1={id(mod1)}, id2={id(mod2)}, id3={id(mod3)}")
""")
ok, out = run_subprocess(check2_code)
results[2] = ok and ("IDENTITY_OK" in out or "MODULE_NONE" in out)
print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — module identity (same object on repeated calls)")

# CHECK 3: Caching reduces call overhead (timing check)
check3_code = textwrap.dedent("""
import time
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import paged_mqa_logits_module

# Warm up
paged_mqa_logits_module()

# Time 1000 cached calls
start = time.perf_counter()
for _ in range(1000):
    paged_mqa_logits_module()
elapsed = time.perf_counter() - start

# Cached calls should complete in < 10ms total (< 10us each)
# Without caching, each call imports a module (~1-10ms), so 1000 calls would take 1-10s
if elapsed < 0.1:  # 100ms generous threshold
    print(f"TIMING_OK: {elapsed*1000:.2f}ms for 1000 calls")
else:
    print(f"TIMING_FAIL: {elapsed*1000:.2f}ms for 1000 calls (expected <100ms)")
""")
ok, out = run_subprocess(check3_code)
results[3] = ok and "TIMING_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — cached call overhead < 100ms/1000 calls")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
