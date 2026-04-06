#!/usr/bin/env python3
"""Test harness for vllm-rocm-nonpow2-blocksize (PR #31380).

Bug: ROCm attention Triton kernels assume power-of-2 block sizes using
bitwise addressing. Qwen3-Next uses block_size=544 (not power of 2),
causing crashes or incorrect results.

Fix: Add PHYSICAL_BLOCK_SIZE constexpr to Triton kernels for generalized
addressing, use BLOCK_SIZE=32 tile with modular arithmetic, and fall back
to triton_reshape_and_cache_flash for non-pow2 cache writes.
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
print("vllm-rocm-nonpow2-blocksize test harness")
print("=" * 60)

# Check 1: Triton cache fallback imported in rocm_attn backend
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
src = inspect.getsource(RocmAttentionImpl)
has_triton_cache = "triton_reshape_and_cache_flash" in src
print(f"HAS_TRITON_CACHE:{has_triton_cache}")
""")
check("Backend has Triton cache fallback for non-pow2 block sizes",
      "HAS_TRITON_CACHE:True" in stdout,
      "no triton_reshape_and_cache_flash in RocmAttentionImpl")

# Check 2: Non-pow2 dispatch logic exists in rocm_attn
# The PR adds a power-of-2 check to branch between native and Triton cache paths
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
src = inspect.getsource(RocmAttentionImpl)
# The fix adds pow2 detection: block_size & (block_size - 1) == 0
has_pow2_check = "block_size - 1" in src or "is_power_of_2" in src.lower() or "pow2" in src.lower()
print(f"HAS_POW2_CHECK:{has_pow2_check}")
""")
check("Backend has power-of-2 block size detection for dispatch",
      "HAS_POW2_CHECK:True" in stdout,
      "no pow2 branching logic found")

# Check 3: PHYSICAL_BLOCK_SIZE exists in the chunked prefill kernel source
# This is a Triton tl.constexpr param, not a Python function parameter
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from pathlib import Path
import importlib.util

# Find the chunked prefill module
spec = importlib.util.find_spec("vllm.attention.ops.chunked_prefill_paged_decode")
if spec is None or spec.origin is None:
    print("MODULE_NOT_FOUND")
else:
    src = Path(spec.origin).read_text()
    has_physical = "PHYSICAL_BLOCK_SIZE" in src
    print(f"HAS_PHYSICAL_BLOCK_SIZE:{has_physical}")
""")
if "MODULE_NOT_FOUND" in stdout:
    check("Chunked prefill kernel has PHYSICAL_BLOCK_SIZE", False, "module not found")
else:
    check("Chunked prefill kernel has PHYSICAL_BLOCK_SIZE",
          "HAS_PHYSICAL_BLOCK_SIZE:True" in stdout,
          "PHYSICAL_BLOCK_SIZE not found in kernel source — non-pow2 addressing not implemented")

# Check 4: PHYSICAL_BLOCK_SIZE exists in prefix prefill kernel source
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("vllm.attention.ops.prefix_prefill")
if spec is None or spec.origin is None:
    print("MODULE_NOT_FOUND")
else:
    src = Path(spec.origin).read_text()
    has_physical = "PHYSICAL_BLOCK_SIZE" in src
    print(f"HAS_PHYSICAL_BLOCK_SIZE:{has_physical}")
""")
if "MODULE_NOT_FOUND" in stdout:
    check("Prefix prefill kernel has PHYSICAL_BLOCK_SIZE", False, "module not found")
else:
    check("Prefix prefill kernel has PHYSICAL_BLOCK_SIZE",
          "HAS_PHYSICAL_BLOCK_SIZE:True" in stdout,
          "PHYSICAL_BLOCK_SIZE not found in prefix prefill kernel")

# Check 5: triton_reshape_and_cache_flash handles 5D head-major layout
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("vllm.attention.ops.triton_reshape_and_cache_flash")
if spec is None or spec.origin is None:
    print("MODULE_NOT_FOUND")
else:
    src = Path(spec.origin).read_text()
    has_head_major = "HEAD_MAJOR" in src or "ndim == 5" in src or "head_major" in src.lower()
    print(f"HAS_HEAD_MAJOR:{has_head_major}")
""")
if "MODULE_NOT_FOUND" in stdout:
    check("Triton cache kernel supports 5D head-major layout", False, "module not found")
else:
    check("Triton cache kernel supports 5D head-major layout",
          "HAS_HEAD_MAJOR:True" in stdout,
          "no head-major / 5D support in triton_reshape_and_cache_flash")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
