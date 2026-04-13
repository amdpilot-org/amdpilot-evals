#!/usr/bin/env python3
"""Behavioral test: KV cache gather kernel handles non-contiguous tensors.

The Triton gather kernel must use actual tensor strides, not hardcoded
contiguous layout assumptions. Pre-fix code used hardcoded pointer arithmetic
that broke for hybrid models (e.g., Jamba) with interleaved KV cache layout.
"""
import subprocess
import sys
import os
import textwrap

NUM_CHECKS = 3
results = {}


def run_subprocess(test_code: str) -> tuple:
    proc = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True, timeout=120, env=os.environ.copy()
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# CHECK 1: Import succeeds
check1_code = textwrap.dedent("""
import torch
if not torch.cuda.is_available() or 'gfx9' not in torch.cuda.get_device_properties(0).gcnArchName:
    print("IMPORT_SKIP")
    exit(1)
try:
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {e}")
    exit(1)
""")
ok, out = run_subprocess(check1_code)
if "IMPORT_SKIP" in out:
    print("SCORE: 0 (IMPORT_SKIP — not ROCm gfx9, auto-FAIL)")
    sys.exit(0)
results[1] = ok and "IMPORT_OK" in out
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — cp_mha_gather_cache import")

# CHECK 2: Contiguous KV cache — gather produces correct output
check2_code = textwrap.dedent("""
import torch
from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

device = torch.device("cuda:0")
num_blocks, page_size, num_heads, head_size = 8, 4, 2, 64
# Create contiguous KV cache [num_blocks, page_size, num_heads, head_size]
key_cache = torch.randn(num_blocks, page_size, num_heads, head_size, device=device, dtype=torch.bfloat16)
value_cache = torch.randn(num_blocks, page_size, num_heads, head_size, device=device, dtype=torch.bfloat16)

# 2 sequences: seq0 uses blocks [0,1], seq1 uses blocks [2,3]
batch = 2
seq_len = page_size * 2  # 8 tokens each
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
cu_seqlens = torch.tensor([0, seq_len, seq_len * 2], dtype=torch.int32, device=device)
total_tokens = seq_len * batch
token_to_batch = torch.cat([torch.full((seq_len,), i, dtype=torch.int32, device=device) for i in range(batch)])
seq_starts = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
k_scales = torch.ones(1, device=device, dtype=torch.float32)
v_scales = torch.ones(1, device=device, dtype=torch.float32)

key_out = torch.empty(total_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
value_out = torch.empty_like(key_out)

cp_mha_gather_cache(
    key_cache, value_cache, key_out, value_out,
    block_tables, k_scales, v_scales,
    cu_seqlens, token_to_batch, seq_starts,
    dequant=False, kv_cache_layout="NHD", total_tokens=total_tokens
)

# Verify: gathered key[0:page_size] should match key_cache[block=0]
ref_key_0 = key_cache[0]  # [page_size, num_heads, head_size]
gathered_key_0 = key_out[:page_size]
if torch.allclose(ref_key_0.float(), gathered_key_0.float(), atol=1e-3):
    print("CONTIGUOUS_OK")
else:
    max_diff = (ref_key_0.float() - gathered_key_0.float()).abs().max().item()
    print(f"CONTIGUOUS_FAIL: max_diff={max_diff}")
""")
ok, out = run_subprocess(check2_code)
results[2] = ok and "CONTIGUOUS_OK" in out
print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — contiguous KV cache gather correctness")

# CHECK 3: Non-contiguous KV cache (simulating hybrid model layout)
check3_code = textwrap.dedent("""
import torch
from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

device = torch.device("cuda:0")
num_blocks, page_size, num_heads, head_size = 8, 4, 2, 64

# Create interleaved [K0,V0,K1,V1,...] layout via as_strided
# Allocate double-sized buffer and stride through it
buf = torch.randn(num_blocks * 2, page_size, num_heads, head_size, device=device, dtype=torch.bfloat16)
# Key cache = even indices [0,2,4,6,...], Value cache = odd indices [1,3,5,7,...]
key_cache = buf[::2]   # Non-contiguous! stride[0] = 2 * page_size * num_heads * head_size
value_cache = buf[1::2]

assert not key_cache.is_contiguous(), "key_cache should be non-contiguous for this test"

batch = 2
seq_len = page_size * 2
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
cu_seqlens = torch.tensor([0, seq_len, seq_len * 2], dtype=torch.int32, device=device)
total_tokens = seq_len * batch
token_to_batch = torch.cat([torch.full((seq_len,), i, dtype=torch.int32, device=device) for i in range(batch)])
seq_starts = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
k_scales = torch.ones(1, device=device, dtype=torch.float32)
v_scales = torch.ones(1, device=device, dtype=torch.float32)

key_out = torch.empty(total_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
value_out = torch.empty_like(key_out)

try:
    cp_mha_gather_cache(
        key_cache, value_cache, key_out, value_out,
        block_tables, k_scales, v_scales,
        cu_seqlens, token_to_batch, seq_starts,
        dequant=False, kv_cache_layout="NHD", total_tokens=total_tokens
    )
    # Verify correctness against reference
    ref_key_0 = key_cache[0]  # block 0 from interleaved layout
    gathered_key_0 = key_out[:page_size]
    if torch.allclose(ref_key_0.float(), gathered_key_0.float(), atol=1e-3):
        print("NONCONTIG_OK")
    else:
        max_diff = (ref_key_0.float() - gathered_key_0.float()).abs().max().item()
        print(f"NONCONTIG_FAIL: max_diff={max_diff}")
except Exception as e:
    print(f"NONCONTIG_CRASH: {e}")
""")
ok, out = run_subprocess(check3_code)
results[3] = ok and "NONCONTIG_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — non-contiguous KV cache gather (hybrid model layout)")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
