#!/usr/bin/env python3
"""Test harness for vllm-rocm-spec-decode-dispatch (PR #32877).

Bug: AITER FlashAttention decode path hardcodes max_seqlen_q=1 when calling
     paged_attention_v1. During speculative decoding, max_query_len > 1, so
     the attention computation produces wrong results.

Expected behavior after fix: When decode metadata has max_query_len > 1, the
     decode path dispatches to unified_attention (or equivalent) instead of
     unconditionally calling paged_attention_v1 with max_seqlen_q=1.
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

# -----------------------------------------------------------------------
# Check 1: Import AITER FA backend
# -----------------------------------------------------------------------
stdout, stderr, rc = run_test("""
import sys
sys.path.insert(0, '/workspace/vllm')
try:
    import importlib
    mod = importlib.import_module("vllm.v1.attention.backends.rocm_aiter_fa")
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
""")

import_ok = "IMPORT:OK" in stdout
check("Import AITER FA backend", import_ok,
      stdout.strip()[:200] if not import_ok else "")

if not import_ok:
    check("Decode dispatches differently for max_query_len > 1", False,
          "import failed")
    check("Multi-token decode does NOT use hardcoded max_seqlen_q=1", False,
          "import failed")
    check("paged_attention_v1 is available via torch.ops", False,
          "import failed")
else:
    # -------------------------------------------------------------------
    # Check 2: Behavioral dispatch test — verify the decode path reads
    #          max_query_len from metadata and dispatches differently for
    #          multi-token queries (max_query_len > 1).
    #
    # Strategy: Patch paged_attention_v1 and unified_attention with
    # tracking wrappers, then invoke the decode forward path twice:
    #   (a) with max_query_len=1 (standard decode)
    #   (b) with max_query_len=4 (speculative decode)
    #
    # Pre-fix: both calls go to paged_attention_v1 with max_seqlen_q=1.
    # Post-fix: call (b) dispatches to unified_attention or passes the
    # actual query length to the attention function.
    # -------------------------------------------------------------------
    stdout2, stderr2, rc2 = run_test("""
import sys, os, json, inspect, traceback
sys.path.insert(0, '/workspace/vllm')

import torch
import importlib

if not torch.cuda.is_available():
    print("NO_GPU")
    sys.exit(0)

device = torch.device("cuda:0")

mod = importlib.import_module("vllm.v1.attention.backends.rocm_aiter_fa")

# Find the implementation class
impl_cls = getattr(mod, "AiterFlashAttentionImpl", None)
if impl_cls is None:
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if isinstance(obj, type) and 'impl' in name.lower() and 'flash' in name.lower():
            impl_cls = obj
            break

if impl_cls is None:
    print("NO_IMPL_CLASS")
    sys.exit(0)

# Track which attention functions are called and with what args
dispatch_log = []

# Patch attention functions to track calls
original_pa_v1 = None
original_unified = None

try:
    original_pa_v1 = torch.ops.aiter.paged_attention_v1
except:
    pass

# Find unified_attention in the module
unified_fn = getattr(mod, "unified_attention", None)
if unified_fn is None:
    try:
        from vllm.v1.attention.backends.utils import unified_attention as uf
        unified_fn = uf
    except:
        pass

def tracking_pa_v1(*args, **kwargs):
    # Track the max_seqlen_q argument
    # paged_attention_v1 signature: (output, query, key_cache, value_cache,
    #   num_kv_heads, scale, block_tables, seq_lens, block_size,
    #   max_seq_len, alibi_slopes, kv_cache_dtype, ...)
    dispatch_log.append({"fn": "paged_attention_v1", "args_count": len(args),
                          "kwargs": list(kwargs.keys())})
    if original_pa_v1 is not None:
        return original_pa_v1(*args, **kwargs)

def tracking_unified(*args, **kwargs):
    dispatch_log.append({"fn": "unified_attention", "args_count": len(args),
                          "kwargs": list(kwargs.keys())})
    if unified_fn is not None:
        return unified_fn(*args, **kwargs)

# Construct the impl
head_size = 128
num_heads = 8
num_kv_heads = 2
scale = head_size ** -0.5

sig = inspect.signature(impl_cls)
ctor_kwargs = {}
for name, param in sig.parameters.items():
    if name == 'self': continue
    nl = name.lower()
    if 'num_head' in nl and 'kv' not in nl: ctor_kwargs[name] = num_heads
    elif 'head_size' in nl or 'head_dim' in nl: ctor_kwargs[name] = head_size
    elif 'scale' in nl: ctor_kwargs[name] = scale
    elif 'kv' in nl and 'head' in nl: ctor_kwargs[name] = num_kv_heads
    elif 'sliding' in nl: ctor_kwargs[name] = (-1, -1)
    elif param.default is not inspect.Parameter.empty: continue
    else: ctor_kwargs[name] = None

try:
    impl = impl_cls(**ctor_kwargs)
except Exception as e:
    print(f"CONSTRUCT_FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

# Now test the decode forward path
# We need to create metadata that has max_query_len info
# The key behavioral test: does the decode path read max_query_len
# and dispatch differently?

# Inspect the forward method to understand what it needs
fwd_sig = inspect.signature(impl.forward)
fwd_params = list(fwd_sig.parameters.keys())

# Create two metadata objects: one with max_query_len=1, one with max_query_len=4
from unittest.mock import MagicMock

def make_metadata(max_query_len):
    meta = MagicMock()
    meta.max_query_len = max_query_len
    # decode_metadata sub-object also needs max_query_len
    meta.decode_metadata = MagicMock()
    meta.decode_metadata.max_query_len = max_query_len

    # Common metadata fields
    meta.num_prefills = 0
    meta.num_prefill_tokens = 0
    meta.num_decode_tokens = 4
    meta.num_decodes = 4
    meta.max_prefill_seq_len = 0
    meta.max_decode_seq_len = 64
    meta.block_tables = torch.zeros(4, 64, dtype=torch.int32, device=device)
    meta.seq_lens = torch.full((4,), 64, dtype=torch.int32, device=device)
    meta.decode_metadata.block_tables = meta.block_tables
    meta.decode_metadata.seq_lens = meta.seq_lens
    meta.decode_metadata.max_seq_len = 64

    # Prefill metadata should be None/empty to force decode path
    meta.prefill_metadata = None
    return meta

batch = 4
seq_len = 64

# Test A: max_query_len=1 (standard decode)
dispatch_log.clear()
meta_single = make_metadata(max_query_len=1)

q_single = torch.randn(batch, num_heads * head_size, device=device, dtype=torch.bfloat16)
output_single = torch.empty_like(q_single)
num_blocks = batch * seq_len
k_cache = torch.randn(num_blocks, 1, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
v_cache = torch.randn(num_blocks, 1, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)

# Patch the dispatch targets
patched = False
try:
    mod.unified_attention = tracking_unified
    if hasattr(mod, 'paged_attention_v1'):
        mod.paged_attention_v1 = tracking_pa_v1
    patched = True
except:
    pass

single_dispatch = None
multi_dispatch = None
single_error = None
multi_error = None

# Build forward kwargs
fwd_kwargs = {}
for name, param in fwd_sig.parameters.items():
    if name == 'self': continue
    nl = name.lower()
    if 'query' in nl or name == 'q': fwd_kwargs[name] = q_single
    elif 'key_cache' in nl: fwd_kwargs[name] = k_cache
    elif 'value_cache' in nl: fwd_kwargs[name] = v_cache
    elif 'output' in nl: fwd_kwargs[name] = output_single
    elif 'attn_metadata' in nl or 'metadata' in nl: fwd_kwargs[name] = meta_single
    elif 'key' in nl or name == 'k': fwd_kwargs[name] = None
    elif 'value' in nl or name == 'v': fwd_kwargs[name] = None
    elif param.default is not inspect.Parameter.empty: continue
    else: fwd_kwargs[name] = None

try:
    impl.forward(**fwd_kwargs)
    if dispatch_log:
        single_dispatch = dispatch_log[-1]["fn"]
    else:
        single_dispatch = "unknown"
except Exception as e:
    single_error = f"{type(e).__name__}:{str(e)[:150]}"

# Test B: max_query_len=4 (speculative decode)
dispatch_log.clear()
meta_multi = make_metadata(max_query_len=4)

# For multi-token decode, query has more tokens per batch element
q_multi = torch.randn(batch * 4, num_heads * head_size, device=device, dtype=torch.bfloat16)
output_multi = torch.empty_like(q_multi)

fwd_kwargs_multi = dict(fwd_kwargs)
fwd_kwargs_multi['query' if 'query' in fwd_kwargs else 'q'] = q_multi
if 'output' in fwd_kwargs_multi:
    fwd_kwargs_multi['output'] = output_multi
for name in fwd_kwargs_multi:
    nl = name.lower()
    if 'attn_metadata' in nl or 'metadata' in nl:
        fwd_kwargs_multi[name] = meta_multi
        break

try:
    impl.forward(**fwd_kwargs_multi)
    if dispatch_log:
        multi_dispatch = dispatch_log[-1]["fn"]
    else:
        multi_dispatch = "unknown"
except Exception as e:
    multi_error = f"{type(e).__name__}:{str(e)[:150]}"

print(f"SINGLE_DISPATCH:{single_dispatch}")
print(f"MULTI_DISPATCH:{multi_dispatch}")
if single_error:
    print(f"SINGLE_ERROR:{single_error}")
if multi_error:
    print(f"MULTI_ERROR:{multi_error}")

# The key behavioral signal: dispatch changed between single and multi
if single_dispatch and multi_dispatch and single_dispatch != multi_dispatch:
    print("DISPATCH_CHANGED:True")
elif multi_dispatch == "unified_attention":
    print("DISPATCH_CHANGED:True")
else:
    print("DISPATCH_CHANGED:False")
""")

    if "NO_GPU" in stdout2:
        check("Decode dispatches differently for max_query_len > 1",
              False, "No GPU available")
        check("Multi-token decode does NOT use hardcoded max_seqlen_q=1",
              False, "No GPU")
    elif "NO_IMPL_CLASS" in stdout2:
        check("Decode dispatches differently for max_query_len > 1",
              False, "AiterFlashAttentionImpl not found")
        check("Multi-token decode does NOT use hardcoded max_seqlen_q=1",
              False, "impl class not found")
    elif "CONSTRUCT_FAIL" in stdout2:
        err = stdout2.split("CONSTRUCT_FAIL:")[1].split("\n")[0].strip()
        check("Decode dispatches differently for max_query_len > 1",
              False, f"construction failed: {err}")
        check("Multi-token decode does NOT use hardcoded max_seqlen_q=1",
              False, "construction failed")
    else:
        # Parse dispatch results
        dispatch_changed = "DISPATCH_CHANGED:True" in stdout2
        multi_dispatch = None
        for line in stdout2.strip().split("\n"):
            if line.startswith("MULTI_DISPATCH:"):
                multi_dispatch = line.split(":")[1].strip()

        # Check 2: dispatch behavior changes for multi-token queries
        # Post-fix: multi-token should go to unified_attention or a
        # different path than single-token
        multi_uses_unified = multi_dispatch == "unified_attention"
        check("Decode dispatches differently for max_query_len > 1",
              dispatch_changed or multi_uses_unified,
              f"single and multi-token decode use same path — "
              f"multi_dispatch={multi_dispatch}")

        # Check 3: multi-token decode should NOT hardcode max_seqlen_q=1
        # If it dispatches to unified_attention, it reads actual query len
        # If it still uses paged_attention_v1, it's hardcoded
        multi_not_hardcoded = (multi_dispatch != "paged_attention_v1"
                                or dispatch_changed)
        check("Multi-token decode does NOT use hardcoded max_seqlen_q=1",
              multi_not_hardcoded,
              f"multi-token decode still uses paged_attention_v1 with "
              f"hardcoded max_seqlen_q=1 — speculative decoding will "
              f"produce wrong results")

    # -------------------------------------------------------------------
    # Check 3: paged_attention_v1 is available via torch.ops
    # -------------------------------------------------------------------
    stdout3, stderr3, rc3 = run_test("""
import sys, torch
try:
    import aiter
    pa_v1 = torch.ops.aiter.paged_attention_v1
    print(f"PA_CALLABLE:{callable(pa_v1)}")
except Exception as e:
    print(f"PA_FAIL:{type(e).__name__}:{str(e)[:200]}")
""")

    if "PA_CALLABLE:True" in stdout3:
        check("paged_attention_v1 is available via torch.ops", True)
    else:
        err = stdout3.strip()[:200] if stdout3.strip() else stderr3.strip()[:200]
        check("paged_attention_v1 is available via torch.ops", False, err)


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
