#!/usr/bin/env python3
"""Test harness for aiter-splitk-buffer-fix eval instance.

Validates that the MoE intermediate buffer is correctly sized when
splitK > 1.  The bug: the buffer was allocated based on (token_num, topk)
but the kernel writes to sorted_token_ids rows which can be larger due to
block_m padding.  This causes out-of-bounds writes.

Tests are behavioral: import the module, call the function at runtime,
and verify the buffer allocation handles padding correctly.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"


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
    return condition


print("=" * 60)
print("aiter-splitk-buffer-fix test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: aiter importable
# ---------------------------------------------------------------------------
import_script = """\
import sys
sys.path.insert(0, '/sgl-workspace/aiter')
try:
    from aiter.fused_moe import ck_moe_stage1, moe_sorting
    import aiter
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:300]}")
"""

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", import_script],
        capture_output=True, text=True, timeout=60, cwd="/workspace",
    )
    stdout = result.stdout
except subprocess.TimeoutExpired:
    stdout = "IMPORT:FAIL:timeout"

if "IMPORT:OK" not in stdout:
    detail = stdout.split("IMPORT:FAIL:")[-1].strip() if "IMPORT:FAIL:" in stdout else "unknown"
    check("aiter fused_moe module importable", False, detail)
    print(f"\nSCORE: 0.0")
    sys.exit(1)
check("aiter fused_moe module importable", True)

# ---------------------------------------------------------------------------
# Checks 1-4: Buffer allocation correctness under splitK
#
# Strategy: call ck_moe_stage1 with splitK>1 parameters, intercept the
# CK kernel call to capture the intermediate buffer shape, and verify
# it is large enough to hold sorted_token_ids rows.
#
# This exercises the actual allocation code path at runtime — not source
# pattern analysis.
# ---------------------------------------------------------------------------

BUFFER_TEST_SCRIPT = r"""
import sys, json, os
sys.path.insert(0, '/sgl-workspace/aiter')

import torch
import aiter
from aiter.fused_moe import ck_moe_stage1, moe_sorting
from aiter import dtypes

torch.manual_seed(42)
device = "cuda"

results = {}

def test_buffer_allocation(token_num, topk, num_experts, block_m, model_dim, inter_dim,
                           test_name):
    """Test that ck_moe_stage1 allocates a correctly-sized buffer for splitK.

    We intercept the CK kernel call to capture the tmp_out shape and verify
    it is >= sorted_token_ids.shape[0].
    """
    # Create synthetic MoE inputs
    hidden_states = torch.randn(token_num, model_dim, dtype=dtypes.bf16, device=device)
    w1 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=dtypes.bf16, device=device)
    w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtypes.bf16, device=device)

    # Create random topk assignments
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device)
    topk_weight = torch.ones(token_num, topk, dtype=dtypes.bf16, device=device) / topk

    # Run moe_sorting to get sorted_token_ids (with padding)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weight, num_experts, model_dim, dtypes.bf16, block_m,
    )

    sorted_len = sorted_ids.shape[0]
    expected_min_rows = sorted_len
    old_rows = token_num * topk

    # Output buffer
    out = torch.zeros(token_num * topk, inter_dim * 2, dtype=dtypes.bf16, device=device)

    # Intercept the CK kernel to capture tmp_out shape
    captured = {}
    original_fwd = aiter.ck_moe_stage1_fwd

    def capturing_fwd(*args, **kwargs):
        # The tmp_out tensor is argument index 6 in the call
        tmp_out = args[6]
        captured['tmp_out_shape'] = list(tmp_out.shape)
        captured['tmp_out_rows'] = tmp_out.shape[0]
        # Don't call the actual kernel (avoids needing FP8 data)

    aiter.ck_moe_stage1_fwd = capturing_fwd

    try:
        ck_moe_stage1(
            hidden_states, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            out, topk, block_m,
            None,  # a1_scale
            None,  # w1_scale
            quant_type=aiter.QuantType.per_1x128,
            splitk=2,
        )
    except Exception as e:
        results[test_name] = {
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "sorted_len": sorted_len,
            "old_rows": old_rows,
        }
        aiter.ck_moe_stage1_fwd = original_fwd
        return
    finally:
        aiter.ck_moe_stage1_fwd = original_fwd

    tmp_out_rows = captured.get('tmp_out_rows', -1)
    buffer_sufficient = tmp_out_rows >= expected_min_rows
    would_overflow = sorted_len > old_rows

    results[test_name] = {
        "sorted_len": sorted_len,
        "old_rows": old_rows,
        "tmp_out_rows": tmp_out_rows,
        "buffer_sufficient": buffer_sufficient,
        "would_overflow_old": would_overflow,
        "error": "",
    }


# Test 1: DeepSeek V3 decode scenario (most critical)
# token_num=1, topk=8, block_m=4, num_experts=8
# Old: 1*8=8 rows. Sorted: 1*8 + 8*4 - 8 = 32 rows. OVERFLOW!
test_buffer_allocation(
    token_num=1, topk=8, num_experts=8, block_m=4,
    model_dim=128, inter_dim=64,
    test_name="deepseek_decode",
)

# Test 2: Small batch decode
# token_num=4, topk=8, block_m=4, num_experts=8
# Old: 4*8=32. Sorted: 4*8 + 8*4 - 8 = 56. OVERFLOW!
test_buffer_allocation(
    token_num=4, topk=8, num_experts=8, block_m=4,
    model_dim=128, inter_dim=64,
    test_name="small_batch_decode",
)

# Test 3: Large batch (no overflow expected)
# token_num=256, topk=2, block_m=4, num_experts=8
# Old: 256*2=512. Sorted: 256*2 + 8*4 - 2 = 542.
# Block padding is small relative to total → still overflows but marginal
test_buffer_allocation(
    token_num=256, topk=2, num_experts=8, block_m=4,
    model_dim=128, inter_dim=64,
    test_name="large_batch",
)

# Test 4: Edge case with many experts
# token_num=1, topk=6, block_m=4, num_experts=64
# Old: 1*6=6. Sorted: 1*6 + 64*4 - 6 = 256. MASSIVE OVERFLOW!
test_buffer_allocation(
    token_num=1, topk=6, num_experts=64, block_m=4,
    model_dim=128, inter_dim=64,
    test_name="many_experts",
)

print(json.dumps(results))
"""

print("\n--- Checks 1-4: buffer allocation under splitK ---")

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", BUFFER_TEST_SCRIPT],
        capture_output=True, text=True, timeout=120, cwd="/workspace",
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
except subprocess.TimeoutExpired:
    stdout = ""
    stderr = "Test timed out after 120s"

import json

parsed = False
try:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            results = json.loads(line)
            parsed = True
            break
except (json.JSONDecodeError, ValueError):
    pass

if not parsed:
    check("GPU buffer test execution", False,
          f"Failed to parse results. stdout: {stdout[:200]}. stderr: {stderr[:200]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("GPU buffer test execution", True)

# Check 1: DeepSeek V3 decode (most critical — 8 rows vs 32 needed)
r = results.get("deepseek_decode", {})
if r.get("error"):
    check("DeepSeek decode (1 token, 8 experts) — buffer correctly sized",
          False, f"Error: {r['error'][:200]}")
else:
    check("DeepSeek decode (1 token, 8 experts) — buffer correctly sized",
          r.get("buffer_sufficient", False),
          f"tmp_out has {r.get('tmp_out_rows')} rows but sorted_token_ids has "
          f"{r.get('sorted_len')} rows — buffer would overflow")

# Check 2: Small batch decode
r = results.get("small_batch_decode", {})
if r.get("error"):
    check("Small batch decode (4 tokens) — buffer correctly sized",
          False, f"Error: {r['error'][:200]}")
else:
    check("Small batch decode (4 tokens) — buffer correctly sized",
          r.get("buffer_sufficient", False),
          f"tmp_out has {r.get('tmp_out_rows')} rows but sorted_token_ids has "
          f"{r.get('sorted_len')} rows")

# Check 3: Large batch
r = results.get("large_batch", {})
if r.get("error"):
    check("Large batch (256 tokens) — buffer correctly sized",
          False, f"Error: {r['error'][:200]}")
else:
    check("Large batch (256 tokens) — buffer correctly sized",
          r.get("buffer_sufficient", False),
          f"tmp_out has {r.get('tmp_out_rows')} rows but sorted_token_ids has "
          f"{r.get('sorted_len')} rows")

# Check 4: Many experts edge case (1 token, 64 experts → massive padding)
r = results.get("many_experts", {})
if r.get("error"):
    check("Many experts (64 experts, 1 token) — buffer correctly sized",
          False, f"Error: {r['error'][:200]}")
else:
    check("Many experts (64 experts, 1 token) — buffer correctly sized",
          r.get("buffer_sufficient", False),
          f"tmp_out has {r.get('tmp_out_rows')} rows but sorted_token_ids has "
          f"{r.get('sorted_len')} rows — {r.get('sorted_len') - r.get('old_rows', 0)} "
          f"rows of padding not accounted for")

# ---------------------------------------------------------------------------
# Check 5: valid_out slicing — verify output is sliced back to token_num*topk
# ---------------------------------------------------------------------------
print("\n--- Check 5: output slicing after kernel ---")

SLICE_TEST_SCRIPT = r"""
import sys, json
sys.path.insert(0, '/sgl-workspace/aiter')

import torch
import aiter
from aiter.fused_moe import ck_moe_stage1, moe_sorting
from aiter import dtypes

torch.manual_seed(42)
device = "cuda"

token_num, topk, num_experts, block_m = 1, 8, 8, 4
model_dim, inter_dim = 128, 64

hidden_states = torch.randn(token_num, model_dim, dtype=dtypes.bf16, device=device)
w1 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=dtypes.bf16, device=device)
w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtypes.bf16, device=device)
topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device)
topk_weight = torch.ones(token_num, topk, dtype=dtypes.bf16, device=device) / topk

sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weight, num_experts, model_dim, dtypes.bf16, block_m,
)

# The output buffer should have exactly token_num * topk rows
out = torch.zeros(token_num * topk, inter_dim * 2, dtype=dtypes.bf16, device=device)

# Track whether silu_and_mul/gelu_and_mul is called with correct shape
captured = {}
original_silu = aiter.silu_and_mul
original_gelu = aiter.gelu_and_mul
original_fwd = aiter.ck_moe_stage1_fwd

def capturing_silu(out_tensor, inp_tensor):
    captured['activation_input_shape'] = list(inp_tensor.shape)
    captured['activation_output_shape'] = list(out_tensor.shape)
    # Don't call actual kernel

def capturing_gelu(out_tensor, inp_tensor):
    captured['activation_input_shape'] = list(inp_tensor.shape)
    captured['activation_output_shape'] = list(out_tensor.shape)

def noop_fwd(*args, **kwargs):
    pass

aiter.silu_and_mul = capturing_silu
aiter.gelu_and_mul = capturing_gelu
aiter.ck_moe_stage1_fwd = noop_fwd

try:
    ck_moe_stage1(
        hidden_states, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m,
        None, None,
        quant_type=aiter.QuantType.per_1x128,
        splitk=2,
    )
except Exception as e:
    captured['error'] = f"{type(e).__name__}: {str(e)[:200]}"

aiter.silu_and_mul = original_silu
aiter.gelu_and_mul = original_gelu
aiter.ck_moe_stage1_fwd = original_fwd

result = {
    'expected_rows': token_num * topk,
    'activation_input_rows': captured.get('activation_input_shape', [-1])[0],
    'has_error': 'error' in captured,
    'error': captured.get('error', ''),
}
print(json.dumps(result))
"""

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", SLICE_TEST_SCRIPT],
        capture_output=True, text=True, timeout=60, cwd="/workspace",
    )
    stdout5 = result.stdout.strip()
except subprocess.TimeoutExpired:
    stdout5 = ""

parsed5 = False
try:
    for line in reversed(stdout5.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result5 = json.loads(line)
            parsed5 = True
            break
except (json.JSONDecodeError, ValueError):
    pass

if parsed5 and not result5.get("has_error"):
    expected = result5.get("expected_rows", -1)
    actual = result5.get("activation_input_rows", -1)
    check("Activation input is sliced to token_num * topk rows",
          actual == expected,
          f"activation received {actual} rows, expected {expected} — "
          f"overflow rows not trimmed before activation function")
elif parsed5:
    check("Activation input is sliced to token_num * topk rows",
          False, result5.get("error", "unknown error"))
else:
    check("Activation input is sliced to token_num * topk rows",
          False, f"Failed to parse. stdout: {stdout5[:200]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
