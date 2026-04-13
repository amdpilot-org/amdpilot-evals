#!/usr/bin/env python3
"""Behavioral test: AITER paged attention fallback for small head sizes.

Models with head_size < 64 must fall back to unified_attention (Triton)
instead of ll4mi (native HIP kernel, requires head_size >= 64). Pre-fix
code crashed with an error from the ll4mi kernel when head_size was too small.
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
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
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
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — AiterFlashAttentionBackend import")

# CHECK 2: Run actual attention forward at head_size=32 (triggers unified_attention fallback)
check2_code = textwrap.dedent("""
import torch, sys, inspect

from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

device = torch.device("cuda:0")
head_size = 32  # Below _MIN_HEAD_SIZE_FOR_LL4MI (64), must trigger fallback
num_heads = 8
num_kv_heads = 2
scale = head_size ** -0.5

# Step 1: Verify backend reports support
if not AiterFlashAttentionBackend.supports_head_size(head_size):
    print("HEADSIZE32_NOT_SUPPORTED")
    sys.exit(0)

# Step 2: Construct the impl and attempt a real forward pass
# This catches the case where supports_head_size returns True
# but the actual forward path crashes (pre-fix: ll4mi kernel fails)
try:
    # Import the impl class
    mod = __import__('vllm.v1.attention.backends.rocm_aiter_fa', fromlist=[''])
    impl_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and 'impl' in attr.lower() and 'flash' in attr.lower():
            impl_cls = obj
            break

    if impl_cls is None:
        print("FORWARD_FAIL: impl class not found")
        sys.exit(0)

    # Construct with head_size=32
    impl = impl_cls(
        num_heads=num_heads, head_size=head_size, scale=scale,
        num_kv_heads=num_kv_heads, alibi_slopes=None,
        sliding_window=None, kv_cache_dtype="auto",
        blocksparse_params=None,
    )

    # Create minimal decode inputs
    batch = 4
    seq_len = 64
    total_tokens = batch  # decode: 1 new token per sequence
    q = torch.randn(total_tokens, num_heads * head_size, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(q)

    # KV cache: [num_blocks, page_size, num_kv_heads, head_size]
    num_blocks = batch * seq_len
    k_cache = torch.randn(num_blocks, 1, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, 1, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)

    block_tables = torch.zeros(batch, seq_len, dtype=torch.int32, device=device)
    for i in range(batch):
        block_tables[i] = torch.arange(i * seq_len, (i + 1) * seq_len, dtype=torch.int32, device=device)
    seq_lens_t = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Attempt forward — pre-fix: ll4mi kernel crashes on head_size=32
    # Post-fix: dispatches to unified_attention (Triton) which handles any head_size
    impl.forward(
        query=q, key=None, value=None,
        key_cache=k_cache, value_cache=v_cache,
        output=output, block_tables=block_tables,
        seq_lens=seq_lens_t,
    )

    # Verify output is valid (not NaN/Inf from kernel failure)
    if torch.isnan(output).any() or torch.isinf(output).any():
        print(f"FORWARD_NAN: {torch.isnan(output).sum().item()} NaN values")
    elif output.shape == q.shape:
        print("FORWARD_OK")
    else:
        print(f"FORWARD_SHAPE: expected {q.shape}, got {output.shape}")

except TypeError as e:
    # Constructor signature mismatch — use signature introspection to adapt
    try:
        insp = inspect
        sig = insp.signature(impl_cls)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            nl = name.lower()
            if 'num_head' in nl and 'kv' not in nl:
                kwargs[name] = num_heads
            elif 'head_size' in nl or 'head_dim' in nl:
                kwargs[name] = head_size
            elif 'scale' in nl:
                kwargs[name] = scale
            elif 'kv' in nl and 'head' in nl:
                kwargs[name] = num_kv_heads
            elif param.default is not insp.Parameter.empty:
                continue
            else:
                kwargs[name] = None
        impl = impl_cls(**kwargs)

        # Re-attempt forward with introspected impl
        try:
            fwd_sig = insp.signature(impl.forward)
            fwd_kwargs = {}
            for name, param in fwd_sig.parameters.items():
                if name == 'self':
                    continue
                nl = name.lower()
                if 'query' in nl or name == 'q':
                    fwd_kwargs[name] = q
                elif 'key_cache' in nl:
                    fwd_kwargs[name] = k_cache
                elif 'value_cache' in nl:
                    fwd_kwargs[name] = v_cache
                elif 'output' in nl:
                    fwd_kwargs[name] = output
                elif 'block_table' in nl:
                    fwd_kwargs[name] = block_tables
                elif 'seq_len' in nl:
                    fwd_kwargs[name] = seq_lens_t
                elif 'key' in nl or name == 'k':
                    fwd_kwargs[name] = None
                elif 'value' in nl or name == 'v':
                    fwd_kwargs[name] = None
                elif param.default is not insp.Parameter.empty:
                    continue
                else:
                    fwd_kwargs[name] = None
            impl.forward(**fwd_kwargs)

            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"FORWARD_NAN: {torch.isnan(output).sum().item()} NaN values")
            else:
                print("FORWARD_OK")
        except Exception as e2:
            print(f"FORWARD_FAIL: cannot reach forward path: {e2}")
    except Exception as e2:
        print(f"FORWARD_FAIL: cannot construct impl: {e2}")

except Exception as e:
    print(f"FORWARD_FAIL: {e}")
""")
ok, out = run_subprocess(check2_code)
results[2] = ok and "FORWARD_OK" in out
print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — head_size=32 attention forward (unified_attention fallback)")

# CHECK 3: head_size=128 (>= 64) — also supported (no regression)
check3_code = textwrap.dedent("""
import torch
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

supported_128 = AiterFlashAttentionBackend.supports_head_size(128)
supported_64 = AiterFlashAttentionBackend.supports_head_size(64)
supported_256 = AiterFlashAttentionBackend.supports_head_size(256)

if supported_128 and supported_64 and supported_256:
    print("LARGE_HEADS_OK")
else:
    print(f"LARGE_HEADS_FAIL: 64={supported_64}, 128={supported_128}, 256={supported_256}")
""")
ok, out = run_subprocess(check3_code)
results[3] = ok and "LARGE_HEADS_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — head_size=64/128/256 supported (no regression)")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
