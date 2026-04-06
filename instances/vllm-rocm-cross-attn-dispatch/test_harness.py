#!/usr/bin/env python3
"""Test harness for vllm-rocm-cross-attn-dispatch (PR #38450).

Bug: ROCM_ATTN and ROCM_AITER_FA advertise ENCODER_DECODER support but compute
cross-attention incorrectly when max_query_len > 1, producing wrong beam search
results for encoder-decoder models (Whisper, BART).
Test: Verify the backends' supports_attn_type returns the correct values.
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
print("vllm-rocm-cross-attn-dispatch test harness")
print("=" * 60)

# Behavioral test: call supports_attn_type() on the actual backend classes.
# Without fix: ENCODER_DECODER returns True. With fix: returns False.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backend import AttentionType

from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend
rocm_dec = RocmAttentionBackend.supports_attn_type(AttentionType.DECODER)
rocm_enc = RocmAttentionBackend.supports_attn_type(AttentionType.ENCODER_DECODER)
print(f"ROCM_ATTN_DECODER:{rocm_dec}")
print(f"ROCM_ATTN_ENCODER_DECODER:{rocm_enc}")

from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
aiter_dec = AiterFlashAttentionBackend.supports_attn_type(AttentionType.DECODER)
aiter_enc = AiterFlashAttentionBackend.supports_attn_type(AttentionType.ENCODER_DECODER)
print(f"AITER_FA_DECODER:{aiter_dec}")
print(f"AITER_FA_ENCODER_DECODER:{aiter_enc}")
""")

if rc != 0:
    check("Import attention backends", False, stderr[:200])
else:
    check("Import attention backends", True)

    check("RocmAttentionBackend supports DECODER",
          "ROCM_ATTN_DECODER:True" in stdout,
          f"got: {[l for l in stdout.splitlines() if 'ROCM_ATTN_DECODER' in l]}")

    check("RocmAttentionBackend does NOT support ENCODER_DECODER",
          "ROCM_ATTN_ENCODER_DECODER:False" in stdout,
          "still advertises ENCODER_DECODER support (broken cross-attention)")

    check("AiterFlashAttentionBackend supports DECODER",
          "AITER_FA_DECODER:True" in stdout,
          f"got: {[l for l in stdout.splitlines() if 'AITER_FA_DECODER' in l]}")

    check("AiterFlashAttentionBackend does NOT support ENCODER_DECODER",
          "AITER_FA_ENCODER_DECODER:False" in stdout,
          "still advertises ENCODER_DECODER support (broken cross-attention)")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
