#!/usr/bin/env python3
"""Test harness for vllm-rocm-attn-blocksize-qwen35.

Verify that the ROCm attention backend supports non-power-of-two block sizes
that are multiples of 16.
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

# Test 1: Check that non-power-of-2 block sizes (multiples of 16) are supported.
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

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
