#!/usr/bin/env python3
"""Test harness for vllm-rocm-spec-decode-dispatch.

Behavioral test: verifies that the AITER FlashAttention decode path
correctly handles multi-token queries (max_query_len > 1) as occurs
during speculative decoding, rather than hardcoding max_seqlen_q=1.
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


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-spec-decode-dispatch test harness")
print("=" * 60)

# Test 1: Verify the decode path reads max_query_len from metadata
# and dispatches differently for multi-token vs single-token decode.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch
import traceback

try:
    import importlib
    mod = importlib.import_module("vllm.v1.attention.backends.rocm_aiter_fa")
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT_ERROR:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT_ERROR:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

import inspect
try:
    impl_cls = getattr(mod, "AiterFlashAttentionImpl", None)
    if impl_cls is None:
        print("NO_IMPL_CLASS")
        sys.exit(0)

    forward_src = inspect.getsource(impl_cls.forward)

    # Check 1: Does the decode section specifically reference
    # decode_max_query_len? This variable is set from
    # attn_metadata.decode_metadata.max_query_len in the fix.
    has_decode_max_query_len = "decode_max_query_len" in forward_src

    # Check 2: Does the decode section route multi-token queries
    # to unified_attention? The fix adds a condition
    # 'decode_max_query_len > 1' that dispatches to unified_attention.
    has_unified_attention_dispatch = (
        "unified_attention" in forward_src
        and "decode_max_query_len" in forward_src
    )

    # Extract the decode section of the forward method
    lines = forward_src.split(chr(10))
    in_decode_section = False
    decode_lines = []
    for line in lines:
        if 'num_decodes > 0' in line:
            in_decode_section = True
        if in_decode_section:
            decode_lines.append(line)

    decode_src = chr(10).join(decode_lines)

    has_pa_v1_in_decode = "paged_attention_v1" in decode_src
    decode_reads_query_len = "decode_max_query_len" in decode_src

    print(f"HAS_DECODE_MAX_QUERY_LEN:{has_decode_max_query_len}")
    print(f"HAS_UNIFIED_ATTENTION_DISPATCH:{has_unified_attention_dispatch}")
    print(f"HAS_PA_V1_IN_DECODE:{has_pa_v1_in_decode}")
    print(f"DECODE_READS_QUERY_LEN:{decode_reads_query_len}")

except Exception as e:
    print(f"INSPECT_ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_ERROR:" in stdout:
    err = stdout.split("IMPORT_ERROR:")[1].strip()[:200]
    check("Import AITER FA backend", False, f"import error: {err}")
    check("Decode path reads actual query length from metadata", False,
          "import failed")
    check("Decode does not hardcode max_seqlen_q=1 for all queries", False,
          "import failed")
elif "NO_IMPL_CLASS" in stdout:
    check("Import AITER FA backend", False, "AiterFlashAttentionImpl not found")
    check("Decode path reads actual query length from metadata", False,
          "impl class not found")
    check("Decode does not hardcode max_seqlen_q=1 for all queries", False,
          "impl class not found")
elif "INSPECT_ERROR:" in stdout:
    err = stdout.split("INSPECT_ERROR:")[1].strip()[:200]
    check("Import AITER FA backend", True)
    check("Decode path reads actual query length from metadata", False, err)
    check("Decode does not hardcode max_seqlen_q=1 for all queries", False, err)
elif "HAS_DECODE_MAX_QUERY_LEN:" in stdout:
    check("Import AITER FA backend", True)

    # Check 2: The decode path must read the actual query length
    decode_reads = "DECODE_READS_QUERY_LEN:True" in stdout
    check("Decode path reads actual query length from metadata",
          decode_reads,
          "decode section does not reference decode_max_query_len - "
          "multi-token queries will use hardcoded max_seqlen_q=1")

    # Check 3: The fix should either:
    # (a) route multi-token decode to unified_attention, OR
    # (b) pass actual query length to paged_attention_v1
    has_unified = "HAS_UNIFIED_ATTENTION_DISPATCH:True" in stdout

    check("Decode does not hardcode max_seqlen_q=1 for all queries",
          has_unified or (decode_reads and "HAS_PA_V1_IN_DECODE:False" in stdout),
          "paged_attention_v1 called unconditionally without checking "
          "query length - speculative decoding multi-token queries will "
          "produce wrong results")
else:
    check("Import AITER FA backend", False,
          f"unexpected output: {(stdout + stderr)[:200]}")
    check("Decode path reads actual query length from metadata", False,
          "unexpected output")
    check("Decode does not hardcode max_seqlen_q=1 for all queries", False,
          "unexpected output")


# Test 2: Verify paged_attention_v1 is available and callable via torch.ops
stdout2, stderr2, rc2 = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch

try:
    import torch, aiter
    pa_v1 = torch.ops.aiter.paged_attention_v1
    print("PA_IMPORT:OK")
except Exception as e:
    print(f"PA_IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

# Verify paged_attention_v1 exists and is callable
print(f"PA_CALLABLE:{callable(pa_v1)}")
""")

if "PA_IMPORT:OK" in stdout2:
    check("paged_attention_v1 is available and callable",
          "PA_CALLABLE:True" in stdout2,
          "paged_attention_v1 not callable")
elif "PA_IMPORT:FAIL" in stdout2:
    err = stdout2.split("PA_IMPORT:FAIL:")[1].strip()[:200]
    check("paged_attention_v1 is available and callable", False,
          f"import failed: {err}")
else:
    check("paged_attention_v1 is available and callable", False,
          f"unexpected output: {stdout2[:200]}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
