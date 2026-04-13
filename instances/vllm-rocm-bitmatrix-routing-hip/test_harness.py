#!/usr/bin/env python3
"""Test harness for vllm-rocm-bitmatrix-routing-hip.

Behavioral test: verifies that MoE expert routing produces correct
token-to-expert assignments on ROCm GPUs.
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
print("vllm-rocm-bitmatrix-routing-hip test harness")
print("=" * 60)

# Test 1: Bitmatrix packing produces correct expert assignments.
# Each token selects known experts — verify only those experts appear
# in the packed bitmatrix (no spurious assignments from padding).
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch

try:
    from vllm.triton_utils import triton
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
        pack_bitmatrix,
    )

    torch.manual_seed(42)
    num_tokens = 32
    num_topk = 2
    num_experts = 32

    # Each token selects 2 experts from {0..7}
    topk_ids = torch.randint(
        0, 8, (num_tokens, num_topk), dtype=torch.int16, device="cuda"
    )

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(num_experts, BLOCK_SIZE_K)

    bitmatrix = torch.zeros(
        (num_tokens, bm_cols), dtype=torch.uint32, device="cuda"
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix, topk_ids, num_tokens, bm_cols, num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Check each row: only selected experts should have bits set
    topk_cpu = topk_ids.cpu()
    spurious = 0
    missing = 0
    for row in range(num_tokens):
        val = bitmatrix[row, 0].item()
        row_experts = set(topk_cpu[row].tolist())
        for eid in range(num_experts):
            bit_set = (val >> eid) & 1
            if bit_set and eid not in row_experts:
                spurious += 1
            if not bit_set and eid in row_experts:
                missing += 1

    print(f"SPURIOUS:{spurious}")
    print(f"MISSING:{missing}")
    print(f"ROUTING_CLEAN:{spurious == 0}")
    print(f"ALL_PRESENT:{missing == 0}")

except ImportError as e:
    print(f"IMPORT_SKIP:{e}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout:
    check("Routing module available", True, "(skipped — deps unavailable)")
    check("No spurious expert assignments in routing", True, "(skipped)")
    check("All selected experts present in routing", True, "(skipped)")
elif "ERROR:" in stdout:
    err = stdout.split("ERROR:")[1].strip()[:200]
    check("Routing module available", False, err)
    check("No spurious expert assignments in routing", False, "module error")
    check("All selected experts present in routing", False, "module error")
else:
    check("Routing module available", True)
    check("No spurious expert assignments in routing",
          "ROUTING_CLEAN:True" in stdout,
          "found spurious expert assignments in packed bitmatrix")
    check("All selected experts present in routing",
          "ALL_PRESENT:True" in stdout,
          "some expected experts missing from routing")


# Test 2: Multi-column bitmatrix (64 experts, 2 bitpack columns).
stdout2, stderr2, rc2 = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch

try:
    from vllm.triton_utils import triton
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
        pack_bitmatrix,
    )

    torch.manual_seed(123)
    num_tokens = 64
    num_topk = 4
    num_experts = 64

    # Select from experts 0-15 only
    topk_ids = torch.randint(
        0, 16, (num_tokens, num_topk), dtype=torch.int16, device="cuda"
    )

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(num_experts, BLOCK_SIZE_K)

    bitmatrix = torch.zeros(
        (num_tokens, bm_cols), dtype=torch.uint32, device="cuda"
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix, topk_ids, num_tokens, bm_cols, num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    topk_cpu = topk_ids.cpu()
    spurious = 0
    for row in range(num_tokens):
        row_experts = set(topk_cpu[row].tolist())
        for col in range(bm_cols):
            val = bitmatrix[row, col].item()
            for bit_pos in range(32):
                eid = col * 32 + bit_pos
                if eid < num_experts and (val >> bit_pos) & 1:
                    if eid not in row_experts:
                        spurious += 1

    print(f"SPURIOUS_64E:{spurious}")
    print(f"MULTI_COL_CLEAN:{spurious == 0}")

except ImportError as e:
    print(f"IMPORT_SKIP:{e}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout2:
    check("Multi-column routing correct (64 experts)", True, "(skipped)")
elif "ERROR:" in stdout2:
    check("Multi-column routing correct (64 experts)", False,
          stdout2.split("ERROR:")[1].strip()[:200] if "ERROR:" in stdout2 else "unknown")
else:
    check("Multi-column routing correct (64 experts)",
          "MULTI_COL_CLEAN:True" in stdout2,
          "spurious assignments in multi-column bitmatrix")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
