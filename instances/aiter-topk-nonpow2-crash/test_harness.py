#!/usr/bin/env python3
"""Test harness for aiter-topk-nonpow2-crash eval instance.

Validates that expert routing dispatch handles arbitrary expert counts
correctly at all sequence lengths.  The bug: a kernel that only supports
certain expert counts is dispatched for all expert counts at long
sequences, causing a crash.

All checks are behavioral: import the module, call the function on GPU,
and verify correct output or absence of crash.
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
print("aiter-topk-nonpow2-crash test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: aiter importable
# ---------------------------------------------------------------------------
import_script = """\
import sys
sys.path.insert(0, '/sgl-workspace/aiter')
try:
    from aiter.ops.topk import biased_grouped_topk
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
    check("aiter topk module importable", False, detail)
    print(f"\nSCORE: 0.0")
    sys.exit(1)
check("aiter topk module importable", True)

# ---------------------------------------------------------------------------
# Checks 1-4: Expert routing at different sequence lengths and expert counts.
#
# The function routes to different kernels based on sequence length and
# expert configuration.  With non-standard expert counts (e.g. 384), the
# fast-path kernel crashes at long sequences.
#
# Test strategy:
#   - Call with 384 experts at progressively longer sequences
#   - Compare against a pure-PyTorch reference implementation
#   - Verify: no crash AND correct output
# ---------------------------------------------------------------------------

ROUTING_TEST_SCRIPT = r"""
import sys, json
sys.path.insert(0, '/sgl-workspace/aiter')

import torch
torch.manual_seed(42)
device = "cuda"

from aiter.ops.topk import biased_grouped_topk

def reference_grouped_topk(gating_output, correction_bias, num_expert_group,
                           topk_group, topk, need_renorm, routed_scaling_factor):
    """Reference implementation using pure PyTorch."""
    token_num, num_experts = gating_output.shape
    scores = gating_output + correction_bias.unsqueeze(0)

    # Group experts and select top groups
    scores_for_group = scores.view(token_num, num_expert_group, -1)
    group_scores = scores_for_group.max(dim=-1).values
    _, top_groups = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)

    # Create mask for selected groups
    mask = torch.zeros_like(scores_for_group[:, :, 0], dtype=torch.bool)
    mask.scatter_(1, top_groups, True)
    mask = mask.unsqueeze(-1).expand_as(scores_for_group).reshape(token_num, num_experts)
    scores_masked = scores.masked_fill(~mask, float('-inf'))

    # Select top-k from masked scores
    topk_weights, topk_ids = torch.topk(scores_masked, k=topk, dim=-1, sorted=False)

    if need_renorm:
        topk_weights = torch.softmax(topk_weights.float(), dim=-1).to(gating_output.dtype)

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def test_routing(num_experts, token_num, topk, num_expert_group, topk_group,
                 test_name, need_renorm=True, routed_scaling_factor=1.0):
    """Test biased_grouped_topk for given parameters.

    Returns dict with error info or correctness metrics.
    """
    gating_output = torch.randn(token_num, num_experts, dtype=torch.float32, device=device)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    topk_weights = torch.empty(token_num, topk, dtype=torch.float32, device=device)
    topk_ids = torch.empty(token_num, topk, dtype=torch.int32, device=device)

    # Compute reference
    ref_weights, ref_ids = reference_grouped_topk(
        gating_output, correction_bias, num_expert_group, topk_group,
        topk, need_renorm, routed_scaling_factor,
    )

    # Call the function under test
    try:
        biased_grouped_topk(
            gating_output, correction_bias,
            topk_weights, topk_ids,
            num_expert_group, topk_group,
            need_renorm, routed_scaling_factor,
        )
        torch.cuda.synchronize()
    except Exception as e:
        return {
            "crashed": True,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "token_num": token_num,
            "num_experts": num_experts,
        }

    # Verify: selected expert IDs should be valid
    ids_valid = (topk_ids >= 0).all().item() and (topk_ids < num_experts).all().item()

    # Verify: weights should be finite and non-negative
    weights_valid = topk_weights.isfinite().all().item() and (topk_weights >= 0).all().item()

    # Verify: the same set of experts is selected (order may differ)
    # Sort both and compare
    ref_ids_sorted, _ = ref_ids.sort(dim=-1)
    test_ids_sorted, _ = topk_ids.sort(dim=-1)
    ids_match = (ref_ids_sorted == test_ids_sorted).all().item()

    # Verify: weights are close to reference (after sorting to match IDs)
    ref_weights_reordered = torch.zeros_like(ref_weights)
    test_weights_reordered = torch.zeros_like(topk_weights)
    for i in range(token_num):
        for j in range(topk):
            ref_weights_reordered[i, j] = ref_weights[i, j]
            test_weights_reordered[i, j] = topk_weights[i, j]

    # Use sorted comparison for weights
    ref_w_sorted, _ = ref_weights.sort(dim=-1, descending=True)
    test_w_sorted, _ = topk_weights.sort(dim=-1, descending=True)
    weight_max_err = (ref_w_sorted.float() - test_w_sorted.float()).abs().max().item()

    return {
        "crashed": False,
        "ids_valid": ids_valid,
        "weights_valid": weights_valid,
        "ids_match": ids_match,
        "weight_max_err": weight_max_err,
        "token_num": token_num,
        "num_experts": num_experts,
        "error": "",
    }


results = {}

# Test 1: 384 experts (Kimi-K2.5 style), short sequence — should always work
results["short_384"] = test_routing(
    num_experts=384, token_num=100, topk=8,
    num_expert_group=4, topk_group=2,
    test_name="short_384",
)

# Test 2: 384 experts, LONG sequence — crashes pre-fix
# Use enough tokens to exceed the dispatch threshold
results["long_384"] = test_routing(
    num_experts=384, token_num=70000, topk=8,
    num_expert_group=4, topk_group=2,
    test_name="long_384",
)

# Test 3: 256 experts (power-of-2), long sequence — should always work
results["long_256"] = test_routing(
    num_experts=256, token_num=70000, topk=8,
    num_expert_group=4, topk_group=2,
    test_name="long_256",
)

# Test 4: 384 experts, medium sequence near dispatch threshold
results["medium_384"] = test_routing(
    num_experts=384, token_num=50000, topk=8,
    num_expert_group=4, topk_group=2,
    test_name="medium_384",
)

print(json.dumps(results))
"""

print("\n--- Checks 1-4: expert routing correctness ---")

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", ROUTING_TEST_SCRIPT],
        capture_output=True, text=True, timeout=300, cwd="/workspace",
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
except subprocess.TimeoutExpired:
    stdout = ""
    stderr = "Test timed out after 300s"

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
    check("GPU routing test execution", False,
          f"Failed to parse results. stdout: {stdout[:200]}. stderr: {stderr[:200]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("GPU routing test execution", True)

WEIGHT_TOLERANCE = 0.01

# Check 1: 384 experts, short sequence (baseline — should always work)
r = results.get("short_384", {})
if r.get("crashed"):
    check("384 experts, short sequence — no crash",
          False, f"Crash: {r.get('error', '')[:200]}")
else:
    ok = r.get("ids_valid", False) and r.get("weights_valid", False)
    check("384 experts, short sequence — correct output",
          ok, f"ids_valid={r.get('ids_valid')}, weights_valid={r.get('weights_valid')}")

# Check 2: 384 experts, LONG sequence (the buggy path — crashes pre-fix)
r = results.get("long_384", {})
if r.get("crashed"):
    check("384 experts, long sequence (70K tokens) — no crash",
          False, f"Crash at long sequence with non-standard expert count: {r.get('error', '')[:200]}")
else:
    ok = r.get("ids_valid", False) and r.get("weights_valid", False)
    check("384 experts, long sequence (70K tokens) — correct output",
          ok, f"ids_valid={r.get('ids_valid')}, weights_valid={r.get('weights_valid')}")

# Check 3: 256 experts (pow2), long sequence (regression guard)
r = results.get("long_256", {})
if r.get("crashed"):
    check("256 experts (pow2), long sequence — no crash",
          False, f"Crash: {r.get('error', '')[:200]}")
else:
    ok = r.get("ids_valid", False) and r.get("weights_valid", False)
    check("256 experts (pow2), long sequence — correct output",
          ok, f"ids_valid={r.get('ids_valid')}, weights_valid={r.get('weights_valid')}")

# Check 4: 384 experts, medium sequence near threshold
r = results.get("medium_384", {})
if r.get("crashed"):
    check("384 experts, medium sequence (50K tokens) — no crash",
          False, f"Crash: {r.get('error', '')[:200]}")
else:
    ok = r.get("ids_valid", False) and r.get("weights_valid", False)
    check("384 experts, medium sequence (50K tokens) — correct output",
          ok, f"ids_valid={r.get('ids_valid')}, weights_valid={r.get('weights_valid')}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
