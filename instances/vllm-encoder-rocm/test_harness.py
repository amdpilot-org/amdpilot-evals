#!/usr/bin/env python3
"""Test harness for vllm-encoder-rocm.

Tests that ROCm attention backends handle all required attention types
for encoder-decoder model support.
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
print("vllm-encoder-rocm test harness")
print("=" * 60)

# Test 1: RocmAttentionBackend.supports_attn_type returns True for ENCODER types
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

for attn_type in ['DECODER', 'ENCODER', 'ENCODER_ONLY']:
    atype = getattr(AttentionType, attn_type)
    try:
        result = RocmAttentionBackend.supports_attn_type(atype)
        print(f"ROCM_{attn_type}:{result}")
    except AttributeError:
        print(f"ROCM_{attn_type}:NO_METHOD")
    except Exception as e:
        print(f"ROCM_{attn_type}:ERROR:{type(e).__name__}")
""")

if rc != 0 and "ROCM_" not in stdout:
    check("Import RocmAttentionBackend", False, stderr[:200])
    check("RocmAttentionBackend supports ENCODER", False, "import failed")
    check("RocmAttentionBackend supports ENCODER_ONLY", False, "import failed")
    check("RocmAttentionBackend supports DECODER", False, "import failed")
else:
    check("Import RocmAttentionBackend", True)
    check("RocmAttentionBackend supports ENCODER",
          "ROCM_ENCODER:True" in stdout,
          "ENCODER not supported or method missing")
    check("RocmAttentionBackend supports ENCODER_ONLY",
          "ROCM_ENCODER_ONLY:True" in stdout,
          "ENCODER_ONLY not supported or method missing")
    check("RocmAttentionBackend supports DECODER",
          "ROCM_DECODER:True" in stdout,
          "DECODER support broken")

# Test 2: RocmAiterUnifiedAttentionBackend.supports_attn_type returns True for ENCODER
stdout2, stderr2, rc2 = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_aiter_unified_attn import RocmAiterUnifiedAttentionBackend

for attn_type in ['DECODER', 'ENCODER', 'ENCODER_ONLY']:
    atype = getattr(AttentionType, attn_type)
    try:
        result = RocmAiterUnifiedAttentionBackend.supports_attn_type(atype)
        print(f"AITER_{attn_type}:{result}")
    except AttributeError:
        print(f"AITER_{attn_type}:NO_METHOD")
    except Exception as e:
        print(f"AITER_{attn_type}:ERROR:{type(e).__name__}")
""")

if rc2 != 0 and "AITER_" not in stdout2:
    check("Import RocmAiterUnifiedAttentionBackend", False, stderr2[:200])
    check("AiterUnified supports ENCODER", False, "import failed")
    check("AiterUnified supports ENCODER_ONLY", False, "import failed")
else:
    check("Import RocmAiterUnifiedAttentionBackend", True)
    check("AiterUnified supports ENCODER",
          "AITER_ENCODER:True" in stdout2,
          "ENCODER not supported or method missing")
    check("AiterUnified supports ENCODER_ONLY",
          "AITER_ENCODER_ONLY:True" in stdout2,
          "ENCODER_ONLY not supported or method missing")

# Test 3: Instantiating RocmAttentionImpl with ENCODER type does NOT raise
# NotImplementedError. Other errors (missing GPU, etc.) are OK.
stdout4, stderr4, rc4 = run_test("""
import sys, math
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
try:
    impl = RocmAttentionImpl(
        num_heads=32, head_size=128, scale=1.0/math.sqrt(128),
        num_kv_heads=8, alibi_slopes=None, sliding_window=(-1, -1),
        kv_cache_dtype="auto", blocksparse_params=None,
        logits_soft_cap=None, attn_type=AttentionType.ENCODER,
        kv_sharing_target_layer_name=None, sinks=None,
    )
    print("INIT_ENCODER:OK")
except NotImplementedError as e:
    print(f"INIT_ENCODER:NOT_IMPLEMENTED:{e}")
except Exception as e:
    # Other errors are acceptable — we only care that NotImplementedError is NOT raised
    print(f"INIT_ENCODER:OTHER_ERROR:{type(e).__name__}")
""")

if "INIT_ENCODER:NOT_IMPLEMENTED" in stdout4:
    check("RocmAttentionImpl accepts ENCODER type (no NotImplementedError)",
          False, "still raises NotImplementedError for ENCODER attention")
else:
    check("RocmAttentionImpl accepts ENCODER type (no NotImplementedError)", True)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
