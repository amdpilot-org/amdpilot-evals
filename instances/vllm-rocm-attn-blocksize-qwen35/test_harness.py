#!/usr/bin/env python3
"""Test harness for vllm-rocm-attn-blocksize-qwen35 (PR #35923).

Bug: ROCm attention backend only routes power-of-2 block sizes to the HIP
cache write path. Qwen3.5 uses block_size=1056 (multiple of 16, not pow2),
which gets misrouted causing nonsensical output.
Test: Verify the backend routes non-pow2-but-16-aligned block sizes correctly
by checking the do_kv_cache_update logic.
"""
import sys
import subprocess

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


def run_test(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-attn-blocksize-qwen35 test harness")
print("=" * 60)

# Check 1: Backend imports
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend, RocmAttentionImpl
print("IMPORT:OK")
""")
check("Import RocmAttentionBackend", "IMPORT:OK" in stdout, stderr[:200])

# Check 2: The forward method handles block sizes that are multiples of 16
# but not powers of 2. Without fix, only power-of-2 goes to native HIP path
# and everything else crashes or produces garbage.
# Behavioral check: inspect the RocmAttentionImpl.forward signature and logic
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

src = inspect.getsource(RocmAttentionImpl)

# The fix changes the gating logic:
# OLD: if is_pow2: → HIP path (only pow2 block sizes use native cache)
# NEW: if block_size in (16, 32): → HIP path (explicit small sizes only)
# This means block_size=1056 goes to Triton path instead of broken HIP path.

# Check the actual gating logic
uses_explicit_sizes = "block_size in (16, 32)" in src or "block_size in {16, 32}" in src
uses_pow2_gate = "if is_pow2:" in src

if uses_explicit_sizes and not uses_pow2_gate:
    print("GATE:EXPLICIT_SIZES")  # Fixed: explicit 16/32 routing
elif uses_pow2_gate:
    print("GATE:POW2_ONLY")  # Broken: pow2-only routing
else:
    print("GATE:UNKNOWN")
""")

gate_ok = "GATE:EXPLICIT_SIZES" in stdout
check("Cache write path uses explicit size routing (not pow2 gate)",
      gate_ok,
      "block_size gating uses is_pow2 — non-pow2 sizes hit wrong code path")

# Check 3: Verify the do_kv_cache_update handles Triton fallback for non-native sizes
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

src = inspect.getsource(RocmAttentionImpl)

# The fix routes non-(16,32) sizes to triton_reshape_and_cache_flash
has_triton_fallback = "triton_reshape_and_cache_flash" in src
print(f"HAS_TRITON_CACHE_FALLBACK:{has_triton_fallback}")
""")

triton_ok = "HAS_TRITON_CACHE_FALLBACK:True" in stdout
check("Triton cache fallback path exists for non-native block sizes",
      triton_ok,
      "no triton_reshape_and_cache_flash fallback — non-standard sizes will fail")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
