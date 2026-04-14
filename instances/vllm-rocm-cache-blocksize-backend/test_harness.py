#!/usr/bin/env python3
"""Behavioral test: attention backend owns block size preference.

The backend's get_preferred_block_size() must return 64 (the value it
actually needs), and supports_block_size() must correctly validate sizes.
Pre-fix code hardcoded block_size in the platform config rather than
delegating to the backend.
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
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import RocmAiterUnifiedAttentionBackend
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
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — RocmAiterUnifiedAttentionBackend import")

# CHECK 2: get_preferred_block_size returns 64 regardless of default
check2_code = textwrap.dedent("""
from vllm.v1.attention.backends.rocm_aiter_unified_attn import RocmAiterUnifiedAttentionBackend as B

test_defaults = [8, 16, 32, 64, 128, 256]
all_ok = True
for d in test_defaults:
    result = B.get_preferred_block_size(d)
    if result != 64:
        print(f"FAIL: get_preferred_block_size({d}) returned {result}, expected 64")
        all_ok = False

if all_ok:
    print("PREFERRED_OK")
else:
    print("PREFERRED_FAIL")
""")
ok, out = run_subprocess(check2_code)
results[2] = ok and "PREFERRED_OK" in out
print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — get_preferred_block_size() returns 64 for all defaults")

# CHECK 3: supports_block_size correctly validates sizes
check3_code = textwrap.dedent("""
from vllm.v1.attention.backends.rocm_aiter_unified_attn import RocmAiterUnifiedAttentionBackend as B

errors = []
# None should be supported (auto-detect)
if not B.supports_block_size(None):
    errors.append("None should be supported")
# Multiples of 16 should be supported
for s in [16, 32, 48, 64, 128]:
    if not B.supports_block_size(s):
        errors.append(f"{s} should be supported (multiple of 16)")
# Non-multiples of 16 should NOT be supported
for s in [15, 17, 31, 33, 65]:
    if B.supports_block_size(s):
        errors.append(f"{s} should NOT be supported (not a multiple of 16)")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    print("VALIDATION_FAIL")
else:
    print("VALIDATION_OK")
""")
ok, out = run_subprocess(check3_code)
results[3] = ok and "VALIDATION_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — supports_block_size() validates correctly")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
