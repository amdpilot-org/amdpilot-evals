#!/usr/bin/env python3
"""Test harness for vllm-rocm-nonpow2-blocksize (PR #31380).

Bug: ROCm attention Triton kernels assume power-of-2 block sizes using
bitwise addressing. Qwen3-Next uses block_size=544 (not power of 2),
causing crashes or incorrect results.
Test: Verify the Triton kernels accept PHYSICAL_BLOCK_SIZE for generalized
addressing, and the backend routes non-pow2 to the Triton cache path.
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

# Check 1: Backend imports
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend
print("IMPORT:OK")
""")
check("ROCm attention backend imports", "IMPORT:OK" in stdout, stderr[:200])

# Check 2: Chunked prefill kernel has PHYSICAL_BLOCK_SIZE for generalized addressing.
# Triton kernel params are tl.constexpr, not visible in Python signatures.
# We check the kernel source file directly.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
path = "/workspace/vllm/vllm/attention/ops/chunked_prefill_paged_decode.py"
with open(path) as f:
    src = f.read()
has_param = "PHYSICAL_BLOCK_SIZE" in src
has_div = "// PHYSICAL_BLOCK_SIZE" in src
has_mod = "% PHYSICAL_BLOCK_SIZE" in src
print(f"CHUNKED_PARAM:{has_param}")
print(f"CHUNKED_DIV:{has_div}")
print(f"CHUNKED_MOD:{has_mod}")
""")
chunked_ok = all(f":{x}" in stdout for x in ["True", "True", "True"]) and "CHUNKED_PARAM:True" in stdout
check("Chunked prefill kernel uses PHYSICAL_BLOCK_SIZE for addressing",
      "CHUNKED_PARAM:True" in stdout and ("CHUNKED_DIV:True" in stdout or "CHUNKED_MOD:True" in stdout),
      "kernel uses bitwise ops only — block_size=544 will crash")

# Check 3: Prefix prefill kernel supports non-pow2 block sizes
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
path = "/workspace/vllm/vllm/attention/ops/prefix_prefill.py"
with open(path) as f:
    src = f.read()
has_param = "PHYSICAL_BLOCK_SIZE" in src
has_div = "// PHYSICAL_BLOCK_SIZE" in src
has_mod = "% PHYSICAL_BLOCK_SIZE" in src
print(f"PREFIX_PARAM:{has_param}")
print(f"PREFIX_DIV:{has_div}")
print(f"PREFIX_MOD:{has_mod}")
""")
check("Prefix prefill kernel supports non-pow2 via PHYSICAL_BLOCK_SIZE",
      "PREFIX_PARAM:True" in stdout and ("PREFIX_DIV:True" in stdout or "PREFIX_MOD:True" in stdout),
      "kernel lacks generalized addressing for non-pow2 block sizes")

# Check 4: ROCm backend detects pow2 vs non-pow2 and routes cache writes
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
src = inspect.getsource(RocmAttentionImpl)
has_pow2 = "is_pow2" in src
has_triton_cache = "triton_reshape_and_cache_flash" in src
print(f"HAS_POW2:{has_pow2}")
print(f"HAS_TRITON_CACHE:{has_triton_cache}")
""")
check("Backend branches on is_pow2 for cache writes",
      "HAS_POW2:True" in stdout and "HAS_TRITON_CACHE:True" in stdout,
      "no pow2 detection — non-pow2 block sizes hit wrong code path")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
