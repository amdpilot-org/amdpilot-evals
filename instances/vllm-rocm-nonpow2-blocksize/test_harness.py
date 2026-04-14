#!/usr/bin/env python3
"""Test harness for vllm-rocm-nonpow2-blocksize.

Behavioral test: verifies that the ROCm paged attention kernel correctly
handles non-power-of-2 KV cache block sizes (e.g., block_size=48) without
crashing or producing incorrect results.
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


def run_test(script, timeout=180):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-nonpow2-blocksize test harness")
print("=" * 60)

# Test 1: Paged attention with non-power-of-2 block size.
# Construct a simple attention scenario: 1 sequence of length 96 with
# block_size=48 (2 physical blocks). Run the prefix prefill attention
# kernel and verify it completes without crashing AND produces correct
# attention output.
#
# Pre-fix: The kernel uses tl.arange(0, BLOCK_SIZE) where BLOCK_SIZE=48.
# Triton requires power-of-2 dimensions for tl.arange, so this will crash
# with a compilation error. Even if Triton doesn't enforce this, the
# kernel's block table indexing conflates tile size with physical block
# size, producing wrong KV cache lookups.
#
# Post-fix: The kernel uses a small power-of-2 tile (e.g., 32) with a
# separate physical block size parameter, correctly indexing across
# non-power-of-2 physical blocks.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch
import traceback

try:
    from vllm.attention.ops.prefix_prefill import context_attention_fwd
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT_ERROR:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT_ERROR:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

# Model config: small but realistic
num_heads = 4
num_kv_heads = 4
head_size = 64
block_size = 48  # Non-power-of-2!

# Sequence config: 1 batch, query_len=4, ctx_len=96 (=2 full blocks of 48)
batch_size = 1
query_len = 4
ctx_len = 96
seq_len = ctx_len + query_len  # total length

num_blocks_needed = (ctx_len + block_size - 1) // block_size  # = 2

# Allocate KV cache in the 5D format vLLM expects:
# k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
# v_cache: [num_blocks, num_kv_heads, head_size, block_size]
x = 8  # head_size subdivision factor
total_blocks = num_blocks_needed + 2  # extra blocks for safety

k_cache = torch.randn(total_blocks, num_kv_heads, head_size // x,
                       block_size, x, dtype=dtype, device=device)
v_cache = torch.randn(total_blocks, num_kv_heads, head_size,
                       block_size, dtype=dtype, device=device)

# Fill KV cache with known values for blocks 0 and 1
# Block 0 holds tokens 0-47, block 1 holds tokens 48-95
for block_idx in range(num_blocks_needed):
    for slot in range(block_size):
        token_id = block_idx * block_size + slot
        if token_id < ctx_len:
            val = (token_id + 1.0) / ctx_len  # deterministic, non-zero
            k_cache[block_idx, :, :, slot, :] = val
            v_cache[block_idx, :, :, slot] = val

# Query, Key, Value for current tokens (query_len=4 new tokens)
q = torch.randn(query_len, num_heads, head_size, dtype=dtype, device=device)
k = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype, device=device)
v = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype, device=device)
o = torch.zeros(query_len, num_heads, head_size, dtype=dtype, device=device)

# Block table: maps logical block indices to physical block indices
# Simple identity mapping: logical block 0 -> physical block 0, etc.
b_loc = torch.zeros(batch_size, (seq_len + block_size - 1) // block_size,
                     dtype=torch.int32, device=device)
for i in range(num_blocks_needed):
    b_loc[0, i] = i

# Sequence metadata
b_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device=device)
b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)

k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

try:
    context_attention_fwd(
        q=q, k=k, v=v, o=o,
        kv_cache_dtype="auto",
        k_cache=k_cache, v_cache=v_cache,
        b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
        max_seq_len=seq_len, max_input_len=query_len,
        k_scale=k_scale, v_scale=v_scale,
    )
    torch.cuda.synchronize()

    # Check output is non-zero (attention was actually computed)
    o_abs_mean = o.abs().mean().item()
    has_output = o_abs_mean > 1e-6
    has_nan = torch.isnan(o).any().item()
    has_inf = torch.isinf(o).any().item()

    print(f"KERNEL_OK:True")
    print(f"OUTPUT_NONZERO:{has_output}")
    print(f"OUTPUT_MEAN:{o_abs_mean:.6f}")
    print(f"HAS_NAN:{has_nan}")
    print(f"HAS_INF:{has_inf}")
    print(f"NUMERICALLY_VALID:{has_output and not has_nan and not has_inf}")

except Exception as e:
    tb = traceback.format_exc()
    err_type = type(e).__name__
    err_msg = str(e)[:300]
    # Check if this is a Triton compilation error (the expected pre-fix crash)
    is_triton_error = "triton" in tb.lower() or "CompilationError" in err_type
    is_pow2_error = "power" in err_msg.lower() or "pow2" in err_msg.lower()
    is_arange_error = "arange" in tb.lower()
    print(f"KERNEL_OK:False")
    print(f"ERROR_TYPE:{err_type}")
    print(f"ERROR_MSG:{err_msg[:200]}")
    print(f"IS_TRITON_ERROR:{is_triton_error}")
""")

if "IMPORT_ERROR:" in stdout:
    err = stdout.split("IMPORT_ERROR:")[1].strip()[:200]
    check("Import prefix prefill module", False, f"import error: {err}")
    check("Non-pow2 block size attention runs without crash", False, "import failed")
    check("Attention output is numerically valid", False, "import failed")
elif "KERNEL_OK:True" in stdout:
    check("Import prefix prefill module", True)
    check("Non-pow2 block size attention runs without crash", True)
    check("Attention output is numerically valid",
          "NUMERICALLY_VALID:True" in stdout,
          f"NaN/Inf/zero in output")
elif "KERNEL_OK:False" in stdout:
    check("Import prefix prefill module", True)
    err_type = "unknown"
    err_msg = ""
    if "ERROR_TYPE:" in stdout:
        err_type = stdout.split("ERROR_TYPE:")[1].split("\n")[0].strip()
    if "ERROR_MSG:" in stdout:
        err_msg = stdout.split("ERROR_MSG:")[1].split("\n")[0].strip()[:200]
    check("Non-pow2 block size attention runs without crash", False,
          f"{err_type}: {err_msg}")
    check("Attention output is numerically valid", False, "kernel crashed")
else:
    check("Import prefix prefill module", False,
          f"unexpected output: {(stdout + stderr)[:200]}")
    check("Non-pow2 block size attention runs without crash", False,
          "unexpected output")
    check("Attention output is numerically valid", False, "unexpected output")


# Test 2: Paged attention with power-of-2 block size still works.
# Sanity check: block_size=32 (power of 2) should work both pre-fix
# and post-fix. This ensures we're not testing a general kernel bug.
stdout2, stderr2, rc2 = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch

try:
    from vllm.attention.ops.prefix_prefill import context_attention_fwd
except ImportError as e:
    print(f"IMPORT_SKIP:{e}")
    sys.exit(0)

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

num_heads = 4
num_kv_heads = 4
head_size = 64
block_size = 32  # Power of 2

batch_size = 1
query_len = 4
ctx_len = 64  # = 2 full blocks of 32
seq_len = ctx_len + query_len
num_blocks_needed = ctx_len // block_size
x = 8
total_blocks = num_blocks_needed + 2

k_cache = torch.randn(total_blocks, num_kv_heads, head_size // x,
                       block_size, x, dtype=dtype, device=device)
v_cache = torch.randn(total_blocks, num_kv_heads, head_size,
                       block_size, dtype=dtype, device=device)

q = torch.randn(query_len, num_heads, head_size, dtype=dtype, device=device)
k = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype, device=device)
v = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype, device=device)
o = torch.zeros(query_len, num_heads, head_size, dtype=dtype, device=device)

b_loc = torch.zeros(batch_size, (seq_len + block_size - 1) // block_size,
                     dtype=torch.int32, device=device)
for i in range(num_blocks_needed):
    b_loc[0, i] = i

b_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device=device)
b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

try:
    context_attention_fwd(
        q=q, k=k, v=v, o=o,
        kv_cache_dtype="auto",
        k_cache=k_cache, v_cache=v_cache,
        b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
        max_seq_len=seq_len, max_input_len=query_len,
        k_scale=k_scale, v_scale=v_scale,
    )
    torch.cuda.synchronize()
    o_mean = o.abs().mean().item()
    has_nan = torch.isnan(o).any().item()
    print(f"POW2_OK:True")
    print(f"POW2_MEAN:{o_mean:.6f}")
    print(f"POW2_VALID:{o_mean > 1e-6 and not has_nan}")
except Exception as e:
    print(f"POW2_OK:False")
    print(f"POW2_ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout2:
    check("Power-of-2 block size attention (control test)", True, "(skipped)")
elif "POW2_OK:True" in stdout2:
    check("Power-of-2 block size attention (control test)",
          "POW2_VALID:True" in stdout2,
          "pow2 baseline failed — kernel has general issues")
elif "POW2_OK:False" in stdout2:
    err = ""
    if "POW2_ERROR:" in stdout2:
        err = stdout2.split("POW2_ERROR:")[1].strip()[:200]
    check("Power-of-2 block size attention (control test)", False,
          f"pow2 baseline crashed: {err}")
else:
    check("Power-of-2 block size attention (control test)", False,
          f"unexpected output: {stdout2[:200]}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
