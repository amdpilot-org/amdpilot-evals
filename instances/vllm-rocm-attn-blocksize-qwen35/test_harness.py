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

# Behavioral test 1: Call supports_block_size with Qwen3.5 block sizes.
# Before fix: returns [16, 32, 544] → block_size 1056 is NOT supported.
# After fix: returns [MultipleOf(16)] → any multiple of 16 is supported.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

# Test Qwen3.5 block sizes (784 and 1056 are multiples of 16 but not pow2)
for bs in [16, 32, 544, 784, 1056]:
    result = RocmAttentionBackend.supports_block_size(bs)
    print(f"SUPPORTS_{bs}:{result}")

# Also check what get_supported_kernel_block_sizes returns
sizes = RocmAttentionBackend.get_supported_kernel_block_sizes()
print(f"SIZES_TYPE:{type(sizes[0]).__name__}")
print(f"NUM_SIZES:{len(sizes)}")
""")

if rc != 0 and "SUPPORTS_" not in stdout:
    check("Import RocmAttentionBackend", False, stderr[:200])
    check("Supports Qwen3.5 block_size 1056", False, "import failed")
    check("Supports Qwen3.5 block_size 784", False, "import failed")
else:
    check("Import RocmAttentionBackend", True)
    check("Supports Qwen3.5 block_size 1056",
          "SUPPORTS_1056:True" in stdout,
          "block_size 1056 rejected — Qwen3.5 will produce garbage output")
    check("Supports Qwen3.5 block_size 784",
          "SUPPORTS_784:True" in stdout,
          "block_size 784 rejected — Qwen3.5 variant will fail")

# Supplementary check: do_kv_cache_update routes non-native sizes to Triton
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
src = inspect.getsource(RocmAttentionImpl)
has_triton_fallback = "triton_reshape_and_cache_flash" in src
print(f"HAS_TRITON_CACHE_FALLBACK:{has_triton_fallback}")
""")
check("Triton cache fallback path exists for non-native block sizes",
      "HAS_TRITON_CACHE_FALLBACK:True" in stdout,
      "no triton_reshape_and_cache_flash fallback — non-standard sizes will fail")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
