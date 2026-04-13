#!/usr/bin/env python3
"""Test harness for aiter-asm-pa-headsize-fix eval instance.

Validates that paged attention produces correct numerical output for
different head sizes. The bug: an unsupported head_size is routed to a
kernel that only handles head_size=128, producing silently incorrect
attention values.
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
print("aiter-asm-pa-headsize-fix test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: aiter importable
# ---------------------------------------------------------------------------
import_script = """\
import sys
sys.path.insert(0, '/sgl-workspace/aiter')
try:
    from aiter.ops.attention import paged_attention_common
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
    check("aiter attention module importable", False, detail)
    print(f"\nSCORE: 0.0")
    sys.exit(1)
check("aiter attention module importable", True)

# ---------------------------------------------------------------------------
# Check 1-3: Paged attention correctness for different head sizes.
#
# Uses many sequences to ensure the high-throughput kernel path is
# exercised (the path that only supports head_size=128).  A reference
# computation gathers KV from the block cache and computes standard
# scaled dot-product attention.
# ---------------------------------------------------------------------------

PA_TEST_SCRIPT = r"""
import sys, math, json
sys.path.insert(0, '/sgl-workspace/aiter')

import torch
torch.manual_seed(42)
device = "cuda"

from aiter.ops.attention import paged_attention_common

def reference_paged_attention(query, k_cache, v_cache, block_tables,
                              context_lens, scale, num_kv_heads):
    """Compute paged attention output using plain PyTorch (reference)."""
    num_seqs, num_heads, head_size = query.shape
    block_size = v_cache.shape[3]
    x = k_cache.shape[4]
    outputs = torch.zeros_like(query, dtype=torch.float32)

    num_queries_per_kv = num_heads // num_kv_heads

    for s in range(num_seqs):
        ctx_len = int(context_lens[s].item())
        bt = block_tables[s].tolist()

        # Gather keys and values from block cache
        k_list, v_list = [], []
        for t in range(ctx_len):
            block_idx = int(bt[t // block_size])
            offset = t % block_size
            # k_cache: [num_blocks, num_kv_heads, head_size//x, block_size, x]
            k_tok = k_cache[block_idx, :, :, offset, :].reshape(num_kv_heads, head_size)
            # v_cache: [num_blocks, num_kv_heads, head_size, block_size]
            v_tok = v_cache[block_idx, :, :, offset]  # [num_kv_heads, head_size]
            k_list.append(k_tok)
            v_list.append(v_tok)

        keys = torch.stack(k_list, dim=0).float()    # [ctx_len, num_kv_heads, head_size]
        values = torch.stack(v_list, dim=0).float()   # [ctx_len, num_kv_heads, head_size]
        q = query[s].unsqueeze(0).float()             # [1, num_heads, head_size]

        # GQA expansion
        if num_queries_per_kv > 1:
            keys = keys.unsqueeze(1).expand(-1, num_queries_per_kv, -1, -1)
            keys = keys.reshape(ctx_len, num_heads, head_size)
            values = values.unsqueeze(1).expand(-1, num_queries_per_kv, -1, -1)
            values = values.reshape(ctx_len, num_heads, head_size)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = scale * torch.einsum("qhd,khd->hqk", q, keys)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, values)
        outputs[s] = out.squeeze(0)

    return outputs.to(query.dtype)


def test_head_size(head_size, num_seqs, num_heads=8, num_kv_heads=8,
                   block_size=16, max_seq_len=64):
    """Run paged attention and compare against reference.

    Returns (max_error, mean_error).
    """
    scale = 1.0 / math.sqrt(head_size)
    max_blocks_per_seq = max_seq_len // block_size
    num_blocks = max_blocks_per_seq * num_seqs

    x = 16 // torch.finfo(torch.float16).bits * 8  # = 8 for fp16

    # Create KV cache
    k_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x,
                          dtype=torch.float16, device=device)
    v_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size,
                          dtype=torch.float16, device=device)

    # Block tables -- each seq uses a contiguous range of blocks
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        for j in range(max_blocks_per_seq):
            block_tables[i, j] = i * max_blocks_per_seq + j

    context_lens = torch.full((num_seqs,), max_seq_len, dtype=torch.int32, device=device)
    query = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)

    # Intermediate buffers for paged_attention_common
    partition_size = 512
    max_num_partitions = (max_seq_len + partition_size - 1) // partition_size
    exp_sums = torch.empty(num_seqs, num_heads, max_num_partitions,
                           dtype=torch.float32, device=device)
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(num_seqs, num_heads, max_num_partitions, head_size,
                          dtype=torch.float32, device=device)

    # Compute reference
    ref = reference_paged_attention(query, k_cache, v_cache, block_tables,
                                    context_lens, scale, num_kv_heads)

    # Run paged_attention_common (exercises the kernel dispatch logic)
    try:
        result = paged_attention_common(
            query, k_cache, v_cache,
            exp_sums, max_logits, tmp_out,
            block_tables, context_lens,
            block_tables.stride(0),
            scale,
            max_qlen=1,
            max_seq_len=max_seq_len,
            kv_cache_dtype="auto",
        )
    except Exception as e:
        return -1.0, -1.0, str(e)

    # Compute error
    max_err = (result.float() - ref.float()).abs().max().item()
    mean_err = (result.float() - ref.float()).abs().mean().item()
    return max_err, mean_err, ""


results = {}

# Test 1: head_size=64 with many seqs (triggers high-throughput dispatch path)
# Pre-fix: wrong kernel selected for head_size=64 → large error
# Post-fix: correct kernel selected → small error
max_err, mean_err, err_msg = test_head_size(64, num_seqs=512)
results["head64_many_seqs"] = {"max_err": max_err, "mean_err": mean_err, "error": err_msg}

# Test 2: head_size=128 with many seqs (should always work correctly)
max_err, mean_err, err_msg = test_head_size(128, num_seqs=512)
results["head128_many_seqs"] = {"max_err": max_err, "mean_err": mean_err, "error": err_msg}

# Test 3: head_size=64 with few seqs (standard dispatch path)
max_err, mean_err, err_msg = test_head_size(64, num_seqs=4)
results["head64_few_seqs"] = {"max_err": max_err, "mean_err": mean_err, "error": err_msg}

# Test 4: head_size=256 with many seqs (another unsupported size)
max_err, mean_err, err_msg = test_head_size(256, num_seqs=512)
results["head256_many_seqs"] = {"max_err": max_err, "mean_err": mean_err, "error": err_msg}

print(json.dumps(results))
"""

print("\n--- Checks 1-4: numerical correctness ---")

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", PA_TEST_SCRIPT],
        capture_output=True, text=True, timeout=300, cwd="/workspace",
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
except subprocess.TimeoutExpired:
    stdout = ""
    stderr = "Test timed out after 300s"

import json

# Tolerance: fp16 attention with these sizes should match within ~0.01
# If the wrong kernel is used, errors will be >> 1.0
TOLERANCE = 0.05

parsed = False
try:
    # Find the JSON line in stdout (last line)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            results = json.loads(line)
            parsed = True
            break
except (json.JSONDecodeError, ValueError):
    pass

if not parsed:
    check("GPU test execution", False,
          f"Failed to parse results. stdout: {stdout[:200]}. stderr: {stderr[:200]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("GPU test execution", True)

# Check 1: head_size=64 with many sequences (the buggy path)
r = results.get("head64_many_seqs", {})
if r.get("error"):
    check("head_size=64, high throughput → correct output",
          False, f"Kernel error: {r['error'][:200]}")
else:
    max_err = r.get("max_err", 999.0)
    check("head_size=64, high throughput → correct output",
          max_err < TOLERANCE and max_err >= 0,
          f"max_err={max_err:.6f} (threshold={TOLERANCE}). "
          "Likely dispatched to a kernel that only supports head_size=128")

# Check 2: head_size=128 (should always work)
r = results.get("head128_many_seqs", {})
if r.get("error"):
    check("head_size=128, high throughput → correct output",
          False, f"Kernel error: {r['error'][:200]}")
else:
    max_err = r.get("max_err", 999.0)
    check("head_size=128, high throughput → correct output",
          max_err < TOLERANCE and max_err >= 0,
          f"max_err={max_err:.6f}")

# Check 3: head_size=64 with few sequences (standard path)
r = results.get("head64_few_seqs", {})
if r.get("error"):
    check("head_size=64, standard path → correct output",
          False, f"Kernel error: {r['error'][:200]}")
else:
    max_err = r.get("max_err", 999.0)
    check("head_size=64, standard path → correct output",
          max_err < TOLERANCE and max_err >= 0,
          f"max_err={max_err:.6f}")

# Check 4: head_size=256 with many sequences (another unsupported size)
r = results.get("head256_many_seqs", {})
if r.get("error"):
    check("head_size=256, high throughput → correct output",
          False, f"Kernel error: {r['error'][:200]}")
else:
    max_err = r.get("max_err", 999.0)
    check("head_size=256, high throughput → correct output",
          max_err < TOLERANCE and max_err >= 0,
          f"max_err={max_err:.6f} (threshold={TOLERANCE}). "
          "Likely dispatched to a kernel that only supports head_size=128")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
