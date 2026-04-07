#!/usr/bin/env python3
"""Behavioral test harness for aiter-bf16-moe-oob-padded.

Verifies that the BF16 CK 2-stage MoE kernel handles all token index
values safely and produces correct output.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

VENV_PYTHON = "/opt/venv/bin/python3"
AITER_PATH = "/sgl-workspace/aiter"

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
    print(msg, flush=True)
    return condition


def run_gpu_script(script_text, timeout=300):
    """Run a GPU test in a subprocess for crash isolation."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(script_text)
        script_path = f.name
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = AITER_PATH
        result = subprocess.run(
            [VENV_PYTHON, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ============================================================================
print("=" * 60)
print("aiter-bf16-moe-oob-padded  BEHAVIORAL test harness")
print("=" * 60)

# --------------------------------------------------------------------------
# Check 0: Target file exists
# --------------------------------------------------------------------------
fused_moe_path = os.path.join(AITER_PATH, "aiter/fused_moe.py")
if not check("fused_moe.py exists", os.path.isfile(fused_moe_path)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# --------------------------------------------------------------------------
# Check 1: GPU is available and aiter imports work
# --------------------------------------------------------------------------
print("\n--- Check 1: GPU and aiter availability ---")

GPU_CANARY = textwrap.dedent("""\
    import sys
    sys.path.insert(0, "/sgl-workspace/aiter")
    import torch
    if not torch.cuda.is_available():
        print("NO_GPU")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    import aiter
    from aiter.fused_moe import moe_sorting
    print("AITER_OK")
""")

rc, out, err = run_gpu_script(GPU_CANARY, timeout=60)
check(
    "GPU available and aiter imports",
    rc == 0 and "AITER_OK" in out,
    f"rc={rc}, stdout={out[:200]}, stderr={err[:200]}",
)
if rc != 0 or "AITER_OK" not in out:
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# --------------------------------------------------------------------------
# Check 2: moe_sorting produces sentinel-padded sorted_ids (precondition)
# --------------------------------------------------------------------------
print("\n--- Check 2: moe_sorting sentinel verification ---")

SENTINEL_CHECK = textwrap.dedent("""\
    import sys, json
    sys.path.insert(0, "/sgl-workspace/aiter")
    import torch
    from aiter.fused_moe import moe_sorting

    num_tokens = 5
    num_experts = 8
    model_dim = 2048
    topk = 2
    block_size = 32
    device = "cuda:0"

    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             dtype=torch.int32, device=device)
    topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32,
                              device=device)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \\
        moe_sorting(topk_ids, topk_weights, num_experts, model_dim,
                    torch.bfloat16, block_size)

    max_id = sorted_ids.max().item()
    num_sentinel = int((sorted_ids >= num_tokens).sum().item())
    total = sorted_ids.numel()
    # Check how many entries within the num_valid_ids range are OOB
    nvi = num_valid_ids[0].item()
    active_slice = sorted_ids[:nvi]
    active_oob = int((active_slice >= num_tokens).sum().item())
    result = {
        "max_id": max_id,
        "num_tokens": num_tokens,
        "num_sentinel": num_sentinel,
        "total": total,
        "has_oob": max_id >= num_tokens,
        "active_oob": active_oob,
        "active_total": nvi,
    }
    print("RESULT:" + json.dumps(result))
""")

rc, out, err = run_gpu_script(SENTINEL_CHECK, timeout=120)
has_sentinels = False
if rc == 0 and "RESULT:" in out:
    import json

    result_line = [l for l in out.splitlines() if l.startswith("RESULT:")][0]
    result = json.loads(result_line[7:])
    has_sentinels = result["has_oob"]
    check(
        "moe_sorting produces sentinel token IDs >= num_tokens",
        has_sentinels,
        f"max_id={result['max_id']}, num_tokens={result['num_tokens']}",
    )
    if has_sentinels:
        print(
            f"    {result['active_oob']}/{result['active_total']} active entries are OOB "
            f"(max_id={result['max_id']}, num_tokens={result['num_tokens']})"
        )
else:
    check(
        "moe_sorting produces sentinel token IDs >= num_tokens",
        False,
        f"Script failed: rc={rc}, stderr={err[:200]}",
    )

# --------------------------------------------------------------------------
# Check 3: OOB sorted_ids cause GPU memory fault (demonstrates danger)
#
# Use raw PyTorch indexing with OOB indices from moe_sorting to prove
# the condition IS dangerous — PyTorch bounds checking catches it.
# --------------------------------------------------------------------------
print("\n--- Check 3: OOB indexing causes crash (danger proof) ---")

OOB_DANGER = textwrap.dedent("""\
    import sys
    sys.path.insert(0, "/sgl-workspace/aiter")
    import torch
    from aiter.fused_moe import moe_sorting

    num_tokens = 5
    model_dim = 2048
    device = "cuda:0"

    hidden_states = torch.randn(num_tokens, model_dim,
                                dtype=torch.bfloat16, device=device)

    topk_ids = torch.randint(0, 8, (num_tokens, 2),
                             dtype=torch.int32, device=device)
    topk_weights = torch.ones(num_tokens, 2, dtype=torch.float32,
                              device=device)
    sorted_ids, _, _, _, _ = moe_sorting(
        topk_ids, topk_weights, 8, model_dim, torch.bfloat16, 32)

    # Convert to long for PyTorch indexing
    indices = sorted_ids.long()
    max_idx = indices.max().item()

    if max_idx >= num_tokens:
        # Try to index with OOB values — should crash or raise error
        try:
            _ = hidden_states[indices]
            torch.cuda.synchronize()
            print("OOB_NO_ERROR")  # Bad — OOB access succeeded
        except (IndexError, RuntimeError) as e:
            print(f"OOB_CAUGHT: {e}")
    else:
        print("NO_OOB_VALUES")
""")

rc3, out3, err3 = run_gpu_script(OOB_DANGER, timeout=120)
# The subprocess should crash (GPU fault) or raise an error
oob_crashes = rc3 != 0 or "OOB_CAUGHT" in out3
check(
    "OOB sorted_ids cause error when used as indices (danger proof)",
    oob_crashes,
    "OOB indexing succeeded without error — unexpected",
)

# --------------------------------------------------------------------------
# Check 4: kernel inputs are within valid bounds (core test)
# --------------------------------------------------------------------------
print("\n--- Check 4: sorted_ids bounded before kernel (core test) ---")

BOUNDS_CHECK = textwrap.dedent("""\
    import sys, json
    sys.path.insert(0, "/sgl-workspace/aiter")
    import torch
    import aiter
    from aiter import ActivationType, QuantType
    import aiter.fused_moe as fm

    results = []

    # Monkey-patch kernel entry points to check sorted_ids bounds
    _orig_ck = fm.ck_moe_stage1
    _orig_cktile = fm.cktile_moe_stage1

    def _checked_ck(hidden_states, w1, w2, sorted_token_ids,
                    sorted_expert_ids, num_valid_ids, out, topk,
                    block_m, a1_scale, w1_scale, **kwargs):
        token_num = hidden_states.shape[0]
        max_id = sorted_token_ids.max().item()
        results.append({
            "kernel": "ck_moe_stage1",
            "max_sorted_id": max_id,
            "token_num": token_num,
            "bounded": max_id < token_num,
        })
        return _orig_ck(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, num_valid_ids, out, topk,
                        block_m, a1_scale, w1_scale, **kwargs)

    def _checked_cktile(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, num_valid_ids, out, topk,
                        block_m, a1_scale, w1_scale, **kwargs):
        token_num = hidden_states.shape[0]
        max_id = sorted_token_ids.max().item()
        results.append({
            "kernel": "cktile_moe_stage1",
            "max_sorted_id": max_id,
            "token_num": token_num,
            "bounded": max_id < token_num,
        })
        return _orig_cktile(hidden_states, w1, w2, sorted_token_ids,
                            sorted_expert_ids, num_valid_ids, out, topk,
                            block_m, a1_scale, w1_scale, **kwargs)

    fm.ck_moe_stage1 = _checked_ck
    fm.cktile_moe_stage1 = _checked_cktile

    # Also patch the functools.partial references in metadata
    # by patching fused_moe_2stages to intercept sorted_ids
    _orig_2stages = fm.fused_moe_2stages
    def _checked_2stages(hidden_states, w1, w2, topk,
                         sorted_ids, sorted_weights,
                         sorted_expert_ids, num_valid_ids,
                         moe_out, *args, **kwargs):
        token_num = hidden_states.shape[0]
        max_id = sorted_ids.max().item()
        results.append({
            "kernel": "fused_moe_2stages_entry",
            "max_sorted_id": max_id,
            "token_num": token_num,
            "bounded": max_id < token_num,
        })
        return _orig_2stages(hidden_states, w1, w2, topk,
                             sorted_ids, sorted_weights,
                             sorted_expert_ids, num_valid_ids,
                             moe_out, *args, **kwargs)
    fm.fused_moe_2stages = _checked_2stages

    # Run fused_moe_ with various batch sizes
    num_experts = 8
    model_dim = 2048
    inter_dim = 256
    topk = 2
    device = "cuda:0"

    w1 = torch.randn(num_experts, inter_dim * 2, model_dim,
                      dtype=torch.bfloat16, device=device)
    w2 = torch.randn(num_experts, model_dim, inter_dim,
                      dtype=torch.bfloat16, device=device)

    for num_tokens in [1, 3, 5, 7, 15, 31]:
        hidden = torch.randn(num_tokens, model_dim,
                              dtype=torch.bfloat16, device=device)
        ids = torch.randint(0, num_experts, (num_tokens, topk),
                            dtype=torch.int32, device=device)
        weights = torch.softmax(
            torch.randn(num_tokens, topk, device=device), dim=-1)

        try:
            fm.fused_moe_(
                hidden, w1, w2, weights, ids,
                activation=ActivationType.Silu.value,
                quant_type=QuantType.No.value,
            )
            torch.cuda.synchronize()
        except Exception as e:
            results.append({"error": str(e)})

    # Report results
    num_checks = sum(1 for r in results if "bounded" in r)
    num_oob = sum(1 for r in results if r.get("bounded") is False)
    # Guard: if no instrumented kernel was reached, the test is inconclusive
    # (agent may have short-circuited fused_moe_ or routed to uninstrumented path)
    if num_checks == 0:
        all_bounded = False
        print("NO_KERNEL_REACHED: True")
    else:
        all_bounded = all(r.get("bounded", False) for r in results if "bounded" in r)
    print(f"BOUNDS_CHECKS: {num_checks}")
    print(f"BOUNDS_OOB: {num_oob}")
    print(f"ALL_BOUNDED: {all_bounded}")
    # Print details for first OOB case
    for r in results:
        if r.get("bounded") is False:
            print(f"OOB_DETAIL: kernel={r['kernel']} max_id={r['max_sorted_id']} "
                  f"token_num={r['token_num']}")
            break
""")

rc4, out4, err4 = run_gpu_script(BOUNDS_CHECK, timeout=300)
if rc4 == 0 and "ALL_BOUNDED:" in out4:
    all_bounded = "ALL_BOUNDED: True" in out4
    # Guard against vacuous pass: if no kernel was instrumented, fail
    no_kernel = "NO_KERNEL_REACHED: True" in out4
    bounds_checks_line = [l for l in out4.splitlines() if l.startswith("BOUNDS_CHECKS:")]
    num_kernel_checks = int(bounds_checks_line[0].split(":")[1].strip()) if bounds_checks_line else 0
    if no_kernel or num_kernel_checks == 0:
        check(
            "sorted_ids are all < num_tokens when reaching kernel",
            False,
            "No instrumented kernel was reached — fused_moe_ may have been short-circuited or routed to uninstrumented path",
        )
    else:
        check(
            "sorted_ids are all < num_tokens when reaching kernel",
            all_bounded,
            "OOB sorted_ids reached the kernel — fix not applied or incomplete",
        )
    # Print details
    for line in out4.splitlines():
        if line.startswith(("BOUNDS_", "OOB_DETAIL", "NO_KERNEL")):
            print(f"    {line}")
else:
    check(
        "sorted_ids are all < num_tokens when reaching kernel",
        False,
        f"rc={rc4}, stderr={err4[-300:] if err4 else 'none'}",
    )

# --------------------------------------------------------------------------
# Check 5: End-to-end correctness — fused_moe_ produces valid output
# --------------------------------------------------------------------------
print("\n--- Check 5: End-to-end fused_moe_ correctness ---")

E2E_TEST = textwrap.dedent("""\
    import sys
    sys.path.insert(0, "/sgl-workspace/aiter")
    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe_

    num_experts = 8
    model_dim = 256
    inter_dim = 64
    topk = 2
    device = "cuda:0"

    torch.manual_seed(42)
    w1 = torch.randn(num_experts, inter_dim * 2, model_dim,
                      dtype=torch.bfloat16, device=device)
    w2 = torch.randn(num_experts, model_dim, inter_dim,
                      dtype=torch.bfloat16, device=device)

    sub_passed = 0
    sub_tested = 0

    # --- Sub-test A: batch=1 deterministic reference ---
    # With a single token, the CK kernel should be deterministic (no
    # scatter race). Run twice and compare to establish a reference.
    sub_tested += 1
    torch.manual_seed(999)
    h1 = torch.randn(1, model_dim, dtype=torch.bfloat16, device=device)
    id1 = torch.tensor([[0, 3]], dtype=torch.int32, device=device)
    wt1 = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device=device)
    r1a = fused_moe_(h1.clone(), w1, w2, wt1, id1,
                      activation=ActivationType.Silu.value,
                      quant_type=QuantType.No.value)
    torch.cuda.synchronize()
    r1b = fused_moe_(h1.clone(), w1, w2, wt1, id1,
                      activation=ActivationType.Silu.value,
                      quant_type=QuantType.No.value)
    torch.cuda.synchronize()
    if torch.allclose(r1a, r1b, atol=1e-4):
        sub_passed += 1
    else:
        print(f"  batch1_ref: non-deterministic max_diff={( r1a - r1b).abs().max().item():.4e}")

    # --- Sub-test B: expert-routing dependency ---
    # Changing which experts are selected must change the output.
    # This rejects implementations that ignore expert routing.
    sub_tested += 1
    id1_alt = torch.tensor([[2, 5]], dtype=torch.int32, device=device)
    r1_alt = fused_moe_(h1.clone(), w1, w2, wt1, id1_alt,
                         activation=ActivationType.Silu.value,
                         quant_type=QuantType.No.value)
    torch.cuda.synchronize()
    if not torch.allclose(r1a, r1_alt, atol=1e-3):
        sub_passed += 1
    else:
        print(f"  routing_dep: output unchanged when expert ids changed")

    # --- Sub-test C: weight dependency ---
    # Changing expert weights must change the output.
    sub_tested += 1
    wt1_alt = torch.tensor([[0.1, 0.9]], dtype=torch.float32, device=device)
    r1_wt = fused_moe_(h1.clone(), w1, w2, wt1_alt, id1,
                        activation=ActivationType.Silu.value,
                        quant_type=QuantType.No.value)
    torch.cuda.synchronize()
    if not torch.allclose(r1a, r1_wt, atol=1e-3):
        sub_passed += 1
    else:
        print(f"  weight_dep: output unchanged when expert weights changed")

    # --- Sub-test D: multi-batch basic properties ---
    for num_tokens in [3, 7, 15, 31, 63]:
        sub_tested += 1
        torch.manual_seed(1000 + num_tokens)
        hidden = torch.randn(num_tokens, model_dim,
                              dtype=torch.bfloat16, device=device)
        ids = torch.randint(0, num_experts, (num_tokens, topk),
                            dtype=torch.int32, device=device)
        weights = torch.softmax(
            torch.randn(num_tokens, topk, device=device), dim=-1)
        try:
            result = fused_moe_(
                hidden.clone(), w1, w2, weights, ids,
                activation=ActivationType.Silu.value,
                quant_type=QuantType.No.value,
            )
            torch.cuda.synchronize()

            ok = True
            reasons = []
            if torch.isnan(result).any().item() or torch.isinf(result).any().item():
                ok = False
                reasons.append("NaN/Inf")
            if ok and result.abs().max().item() < 1e-6:
                ok = False
                reasons.append("all-zeros")
            if ok:
                sub_passed += 1
            else:
                print(f"  batch={num_tokens}: FAIL ({', '.join(reasons)})")
        except Exception as e:
            print(f"  batch={num_tokens}: {e}")

    print(f"E2E_RESULT: {sub_passed}/{sub_tested}")
""")

rc5, out5, err5 = run_gpu_script(E2E_TEST, timeout=300)
if rc5 == 0 and "E2E_RESULT:" in out5:
    parts = out5.split("E2E_RESULT:")[1].strip().split("/")
    e2e_passed = int(parts[0])
    e2e_total = int(parts[1])
    check(
        f"fused_moe_ produces valid output for all batch sizes ({e2e_passed}/{e2e_total})",
        e2e_passed == e2e_total,
        f"Only {e2e_passed}/{e2e_total} passed",
    )
else:
    check(
        "fused_moe_ produces valid output for all batch sizes",
        False,
        f"rc={rc5}, stderr={err5[-300:] if err5 else 'none'}",
    )

# --------------------------------------------------------------------------
# Summary — weighted scoring
# --------------------------------------------------------------------------
print()

# Weighted scoring: Check 4 = 60 pts, others = 10 pts each (40 pts total)
CORE_CHECK_IDX = 4  # 0-indexed: Check 4 is the 5th check() call (index 4)
check_weights = []
for i in range(checks_total):
    check_weights.append(60.0 if i == CORE_CHECK_IDX else 40.0 / max(checks_total - 1, 1))

# Reconstruct pass/fail per check from the global counters
# (we tracked them in order, so check i passed iff checks_passed > i at that point)
# Simpler: just use the variables we have
weighted_score = 0.0
# We know checks 0..CORE_CHECK_IDX-1 all passed (they're prerequisites)
# Check CORE_CHECK_IDX passed iff all_bounded (from Check 4 result)
# Checks after CORE_CHECK_IDX: we know the total passed count
prereq_count = CORE_CHECK_IDX  # checks before the core check
post_count = checks_total - CORE_CHECK_IDX - 1  # checks after core check
post_passed = checks_passed - prereq_count - (1 if "ALL_BOUNDED: True" in (out4 if rc4 == 0 else "") else 0)

for i in range(prereq_count):
    weighted_score += check_weights[i]  # all prereqs passed
if rc4 == 0 and "ALL_BOUNDED: True" in out4:
    weighted_score += 60.0  # core check passed
if post_count > 0:
    weighted_score += (post_passed / post_count) * sum(check_weights[CORE_CHECK_IDX + 1:])

weighted_score = min(100.0, max(0.0, weighted_score))
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {weighted_score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
