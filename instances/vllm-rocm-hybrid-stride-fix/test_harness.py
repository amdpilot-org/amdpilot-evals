#!/usr/bin/env python3
"""Test harness for vLLM hybrid model KV cache stride corruption on ROCm.

Tests (behavioral):
  1. Import the rocm_aiter_fa backend and verify cp_mha_gather_cache exists.
  2. Call cp_mha_gather_cache with non-contiguous (interleaved) KV cache —
     output must match values from the correct block (not the adjacent one).
  3. Call cp_mha_gather_cache with contiguous KV cache (control) —
     output must be correct (both pre-fix and post-fix should pass).
  4. Source verification: kernel receives actual tensor strides, not hardcoded
     pointer arithmetic.
"""

import os
import subprocess
import sys
import json

_PY = "/usr/bin/python3"


def run_check(name, script, timeout=180):
    """Run a subprocess check and return (pass, detail)."""
    proc = subprocess.run(
        [_PY, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": "/workspace/vllm"},
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        return False, f"Exit code {proc.returncode}: {stderr[-500:]}"

    # Parse structured output
    try:
        result = json.loads(stdout.split("\n")[-1])
        return result.get("pass", False), result.get("detail", stdout[-200:])
    except (json.JSONDecodeError, IndexError):
        return True, stdout[-200:]


def main():
    print("=" * 60)
    print("vLLM Hybrid Model KV Cache Stride Test")
    print("=" * 60)

    checks = []

    # Check 1: Import backend and verify cp_mha_gather_cache exists
    print("\n[Check 1] Import rocm_aiter_fa backend...")
    script1 = """
import sys, json
sys.path.insert(0, "/workspace/vllm")
try:
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache
    import inspect
    sig = inspect.signature(cp_mha_gather_cache)
    params = list(sig.parameters.keys())
    print(json.dumps({"pass": True, "detail": f"cp_mha_gather_cache OK, params: {params[:6]}..."}))
except ImportError as e:
    print(json.dumps({"pass": False, "detail": f"Import failed: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("import", script1)
    checks.append({"name": "import_backend", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 2: Interleaved (non-contiguous) KV cache gather
    # Core test: with pre-fix hardcoded strides, the kernel reads from wrong
    # blocks. Post-fix uses actual tensor strides and reads correctly.
    #
    # Setup: interleaved buffer [K0, V0, K1, V1, ...] with distinct values.
    # key_cache = interleaved[0::2] (non-contiguous, stride0 doubled).
    # block_tables points to block 1. Pre-fix reads V0 instead of K1.
    print("\n[Check 2] Interleaved KV gather correctness...")
    script2 = """
import torch, sys, json
sys.path.insert(0, "/workspace/vllm")
torch.cuda.set_device(0)
device = "cuda:0"

try:
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    num_blocks = 4
    page_size = 8
    num_heads = 2
    head_dim = 64
    num_tokens = 4

    # Build interleaved buffer: [K0, V0, K1, V1, K2, V2, K3, V3]
    interleaved = torch.zeros(
        num_blocks * 2, page_size, num_heads, head_dim,
        dtype=torch.bfloat16, device=device
    )
    for i in range(num_blocks):
        interleaved[i * 2].fill_(float(i + 1))        # K_i = i+1
        interleaved[i * 2 + 1].fill_(float(-(i + 1))) # V_i = -(i+1)

    # Non-contiguous views (stride0 is 2x the contiguous stride)
    key_cache = interleaved[0::2]    # [4, 8, 2, 64]
    value_cache = interleaved[1::2]  # [4, 8, 2, 64]

    # Sanity: confirm non-contiguity
    expected_contig_stride0 = page_size * num_heads * head_dim
    actual_stride0 = key_cache.stride(0)
    assert actual_stride0 == 2 * expected_contig_stride0, (
        f"Expected doubled stride, got {actual_stride0} vs 2*{expected_contig_stride0}"
    )

    # 1 batch, 4 tokens, reading from block 1
    block_tables = torch.tensor([[1]], dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    token_to_batch = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    seq_starts = torch.tensor([0], dtype=torch.int32, device=device)

    key_out = torch.zeros(num_tokens, num_heads, head_dim,
                          dtype=torch.bfloat16, device=device)
    value_out = torch.zeros(num_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device=device)

    k_scales = torch.ones(1, dtype=torch.float32, device=device)
    v_scales = torch.ones(1, dtype=torch.float32, device=device)

    cp_mha_gather_cache(
        key_cache, value_cache,
        key_out, value_out,
        block_tables,
        k_scales, v_scales,
        cu_seqlens_kv, token_to_batch, seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=num_tokens,
    )

    # Expected: K1=2.0, V1=-2.0
    # Pre-fix (wrong stride): K reads V0=-1.0, V reads K1=2.0
    # Post-fix (correct stride): K reads K1=2.0, V reads V1=-2.0
    k_mean = key_out.float().mean().item()
    v_mean = value_out.float().mean().item()

    k_correct = abs(k_mean - 2.0) < 0.1
    v_correct = abs(v_mean - (-2.0)) < 0.1

    ok = k_correct and v_correct
    detail = f"k_mean={k_mean:.3f} (expect 2.0), v_mean={v_mean:.3f} (expect -2.0)"
    print(json.dumps({"pass": ok, "detail": detail}))

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(json.dumps({"pass": False, "detail": f"Error: {e} | {tb[-300:]}"}))
    sys.exit(1)
"""
    passed, detail = run_check("interleaved", script2)
    checks.append({"name": "interleaved_kv_gather", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 3: Contiguous KV cache gather (control test)
    # Both pre-fix and post-fix should produce correct results with contiguous
    # layout, since the hardcoded arithmetic happens to be correct for that case.
    print("\n[Check 3] Contiguous KV gather (control)...")
    script3 = """
import torch, sys, json
sys.path.insert(0, "/workspace/vllm")
torch.cuda.set_device(0)
device = "cuda:0"

try:
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    num_blocks = 4
    page_size = 8
    num_heads = 2
    head_dim = 64
    num_tokens = 4

    # Contiguous KV cache (standard layout)
    key_cache = torch.zeros(num_blocks, page_size, num_heads, head_dim,
                            dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros(num_blocks, page_size, num_heads, head_dim,
                              dtype=torch.bfloat16, device=device)
    for i in range(num_blocks):
        key_cache[i].fill_(float(i + 1))
        value_cache[i].fill_(float(-(i + 1)))

    # Same setup: read from block 1
    block_tables = torch.tensor([[1]], dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    token_to_batch = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    seq_starts = torch.tensor([0], dtype=torch.int32, device=device)

    key_out = torch.zeros(num_tokens, num_heads, head_dim,
                          dtype=torch.bfloat16, device=device)
    value_out = torch.zeros(num_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device=device)

    k_scales = torch.ones(1, dtype=torch.float32, device=device)
    v_scales = torch.ones(1, dtype=torch.float32, device=device)

    cp_mha_gather_cache(
        key_cache, value_cache,
        key_out, value_out,
        block_tables,
        k_scales, v_scales,
        cu_seqlens_kv, token_to_batch, seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=num_tokens,
    )

    # Expected: K=2.0 (block 1 key), V=-2.0 (block 1 value)
    k_mean = key_out.float().mean().item()
    v_mean = value_out.float().mean().item()

    k_correct = abs(k_mean - 2.0) < 0.1
    v_correct = abs(v_mean - (-2.0)) < 0.1

    ok = k_correct and v_correct
    detail = f"k_mean={k_mean:.3f} (expect 2.0), v_mean={v_mean:.3f} (expect -2.0)"
    print(json.dumps({"pass": ok, "detail": detail}))

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(json.dumps({"pass": False, "detail": f"Error: {e} | {tb[-300:]}"}))
    sys.exit(1)
"""
    passed, detail = run_check("contiguous", script3)
    checks.append({"name": "contiguous_kv_control", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 4: Source verification — strides passed to kernel
    print("\n[Check 4] Kernel receives actual tensor strides...")
    script4 = """
import sys, json, inspect
sys.path.insert(0, "/workspace/vllm")

try:
    from vllm.v1.attention.backends import rocm_aiter_fa

    src = inspect.getsource(rocm_aiter_fa)

    # The fix adds stride parameters to the kernel call and computes
    # them from the actual tensor via .stride()
    has_k_stride = "k_cache_stride" in src or "k_strides" in src
    has_v_stride = "v_cache_stride" in src or "v_strides" in src
    computes_strides = ".stride()" in src

    ok = has_k_stride and has_v_stride and computes_strides
    detail = f"k_stride={has_k_stride}, v_stride={has_v_stride}, computes_stride={computes_strides}"
    print(json.dumps({"pass": ok, "detail": detail}))

except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Error: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("stride_source", script4)
    checks.append({"name": "stride_params_in_kernel", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Score
    total = len(checks)
    passed_count = sum(1 for c in checks if c["pass"])

    print(f"\n--- Results ---")
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['detail'][:100]}")

    if total == 0:
        print("\nNo checks completed.")
        print("SCORE: 0.0")
        return

    score = passed_count / total * 100.0
    print(f"\n{passed_count}/{total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
