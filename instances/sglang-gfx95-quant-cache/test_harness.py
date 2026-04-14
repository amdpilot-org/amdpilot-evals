#!/usr/bin/env python3
"""Test harness for sglang-gfx95-quant-cache.

Verifies that quantization format detection in the decoder layer is amortized
(performed at most once) rather than on every forward() call.  Amortization is
proven behaviorally by instrumenting weight.dtype accesses and capturing the
quant_format argument flowing to prepare_attn — no source code analysis.
"""
import os
import subprocess
import sys
import textwrap

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
    return condition


print("=" * 60)
print("sglang-gfx95-quant-cache test harness")
print("=" * 60)

# -------------------------------------------------------------------
# Check 1: Module imports successfully
# -------------------------------------------------------------------
result = subprocess.run(
    [sys.executable, "-c",
     "from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer; print('OK')"],
    capture_output=True, text=True, timeout=60,
    cwd="/workspace",
)

if "OK" not in result.stdout:
    check("Import decoder layer class", False,
          "IMPORT_SKIP — auto-FAIL")
    check("Quant format detection is amortized (dtype-access proof)", False,
          "import failed")
    check("Module structure intact", False,
          "import failed")
    print()
    print(f"Results: {checks_passed}/{checks_total}")
    print(f"SCORE: 0.0")
    sys.exit(0)

check("Import decoder layer class", True)

# -------------------------------------------------------------------
# Check 2: Amortization — dtype-access counting + quant_format capture
#
# Strategy:
# 1. Bypass __init__ via __new__ to avoid heavy sglang server
#    dependencies. This tests the REAL forward() method code.
# 2. Patch module-level _is_gfx95_supported = True so the detection
#    branch is entered.
# 3. Set up self_attn with fused_qkv_a_proj_with_mqa whose
#    weight.dtype property counts accesses and returns torch.uint8
#    (triggers "mxfp4" detection path).
# 4. Wrap layer_communicator.prepare_attn to capture quant_format arg.
# 5. Stub downstream compute (self_attn.__call__, mlp.__call__,
#    prepare_mlp, postprocess_layer, etc.) to cheap pass-throughs.
#    CRITICAL: do NOT replace self_attn wholesale — only stub its
#    forward method, keeping fused_qkv_a_proj_with_mqa intact so
#    the detection code path can read weight.dtype.
# 6. Drive N=50 forward() calls with the CORRECT real signature:
#    forward(positions, hidden_states, forward_batch, residual,
#            zero_allocator)
# 7. PASS iff ALL THREE:
#    (a) forward() succeeds (at least N/2 calls complete)
#    (b) weight.dtype accessed <= 1 time across all N calls
#    (c) quant_format values captured at prepare_attn are non-empty
#        and stable (same value every call)
#
# Pre-fix: every forward() re-reads weight.dtype → count = N → FAIL
# Post-fix (any caching strategy): count 0 or 1 → PASS
# Dead-attribute gaming: doesn't reduce dtype accesses → still FAIL
# Dead-method gaming: doesn't reduce dtype accesses → still FAIL
# Dead cached_property: doesn't reduce dtype accesses → still FAIL
# -------------------------------------------------------------------
amort_script = textwrap.dedent(r'''
import sys, types, re
import torch
import torch.nn as nn

# ---- Import the class under test ----
try:
    import sglang.srt.models.deepseek_v2 as ds_mod
    from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
except Exception as e:
    print(f"IMPORT_FAIL:{e}")
    sys.exit(1)

# ---- Patch _is_gfx95_supported so detection branch is entered ----
ds_mod._is_gfx95_supported = True

# ---- dtype-counting weight proxy ----
class _DtypeCountingWeight:
    """Proxy weight object whose .dtype property counts accesses."""
    def __init__(self, target_dtype=torch.uint8):
        self._target_dtype = target_dtype
        self.access_count = 0
    @property
    def dtype(self):
        self.access_count += 1
        return self._target_dtype

# ---- Build a minimal config ----
from unittest.mock import MagicMock
import json, os, tempfile

min_cfg = {
    "model_type": "deepseek_v2",
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "moe_intermediate_size": 1408,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 102400,
    "max_position_embeddings": 4096,
    "rope_theta": 10000,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
}

config = None
try:
    from transformers import AutoConfig
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    )
    json.dump(min_cfg, tf)
    tf.close()
    try:
        config = AutoConfig.from_pretrained(tf.name, trust_remote_code=True)
    finally:
        os.unlink(tf.name)
except Exception:
    pass

# ---- Construct the layer ----
# Attempt 1: Normal construction with mocked server dependencies.
# This exercises __init__ (including any init-time caching a fix adds).
layer = None
construction_method = None

if config is not None:
    try:
        # Mock server-level dependencies
        ds_mod.get_global_server_args = lambda: MagicMock()
        try:
            import sglang.srt.server_args as _sa_mod
            _sa_mod.get_global_server_args = lambda: MagicMock()
            _sa_mod._global_server_args = MagicMock()
        except Exception:
            pass
        try:
            import sglang.srt.speculative.spec_info as _spec_mod
            _spec_mod.SpeculativeAlgorithm.from_string = classmethod(
                lambda cls, name: _spec_mod.SpeculativeAlgorithm.NONE
            )
        except Exception:
            pass
        with torch.device("meta"):
            try:
                layer = DeepseekV2DecoderLayer(config, layer_idx=0)
            except TypeError:
                layer = DeepseekV2DecoderLayer(config, 0)
        construction_method = "normal"
    except Exception:
        pass

# Attempt 2: __new__ bypass (avoids ALL __init__ dependencies).
if layer is None:
    try:
        layer = DeepseekV2DecoderLayer.__new__(DeepseekV2DecoderLayer)
        nn.Module.__init__(layer)
        construction_method = "bypass"
    except Exception as e:
        print(f"CONSTRUCT_FAIL:{e}")
        sys.exit(1)

print(f"CONSTRUCTION:{construction_method}")

# ---- Set up detection surface + stubs ----
# For both construction paths, we need:
#   - self_attn with fused_qkv_a_proj_with_mqa.weight.dtype counting proxy
#   - layer_communicator with prepare_attn capturing quant_format
#   - mlp/other compute stubbed to passthroughs
weight_proxy = _DtypeCountingWeight(torch.uint8)

if construction_method == "normal" and hasattr(layer, 'self_attn'):
    # Normal construction: self_attn already exists. Install proxy on
    # the existing submodule chain, preserving the real object.
    attn = layer.self_attn
    proj = getattr(attn, 'fused_qkv_a_proj_with_mqa', None)
    if proj is not None and hasattr(proj, 'weight'):
        proj.weight = weight_proxy
    else:
        # Create the submodule chain if missing
        mock_proj = types.SimpleNamespace(weight=weight_proxy)
        attn.fused_qkv_a_proj_with_mqa = mock_proj
    # Stub self_attn.forward only (keep module object intact)
    attn.forward = lambda *a, **kw: kw.get(
        'hidden_states',
        next((x for x in a if isinstance(x, torch.Tensor)), torch.zeros(1)),
    )
else:
    # Bypass construction: create mock self_attn with detection surface
    mock_attn = nn.Module()
    mock_proj = types.SimpleNamespace(weight=weight_proxy)
    mock_attn.fused_qkv_a_proj_with_mqa = mock_proj
    mock_attn.forward = lambda *a, **kw: kw.get(
        'hidden_states',
        next((x for x in a if isinstance(x, torch.Tensor)), torch.zeros(1)),
    )
    layer.self_attn = mock_attn

# layer_communicator: capture quant_format, pass through tensors
quant_format_log = []
mock_comm = types.SimpleNamespace()
mock_comm.prepare_attn = lambda hs, res, fb, quant_format="", **kw: (
    quant_format_log.append(quant_format)
    or (hs, res if res is not None else hs)
)
mock_comm.prepare_mlp = lambda hs, res, fb, **kw: (hs, res)
mock_comm.should_fuse_mlp_allreduce_with_next_layer = lambda fb: False
mock_comm.should_use_reduce_scatter = lambda fb: False
mock_comm.postprocess_layer = lambda hs, res, fb, **kw: (hs, res)
layer.layer_communicator = mock_comm

# mlp: return first tensor arg
mock_mlp = nn.Module()
mock_mlp.forward = lambda *a, **kw: next(
    (x for x in a if isinstance(x, torch.Tensor)), torch.zeros(1),
)
layer.mlp = mock_mlp

if not hasattr(layer, 'nsa_enable_prefill_cp'):
    layer.nsa_enable_prefill_cp = False
if not hasattr(layer, 'layer_scatter_modes'):
    layer.layer_scatter_modes = None
layer.eval()

# ---- Create mock inputs matching real forward() signature ----
#   forward(positions, hidden_states, forward_batch, residual, zero_allocator)
hidden_size = 2048
hidden = torch.randn(4, hidden_size)
positions = torch.zeros(4, dtype=torch.long)
residual = torch.zeros_like(hidden)

# Lightweight sentinel that absorbs any attribute access / method call
class _AnyMock:
    def __getattr__(self, name):
        return _AnyMock()
    def __call__(self, *a, **kw):
        return _AnyMock()
    def __bool__(self):
        return False
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0

forward_batch = _AnyMock()
zero_allocator = _AnyMock()

# ---- Drive N forward() calls with correct signature ----
N = 50
fwd_ok = 0
init_cache_evidence = False

for i in range(N):
    try:
        result = layer.forward(
            positions, hidden, forward_batch, residual, zero_allocator,
        )
        fwd_ok += 1
        if isinstance(result, tuple) and len(result) >= 2:
            hidden, residual = result[0], result[1]
    except AttributeError as e:
        if not init_cache_evidence and construction_method == "bypass":
            # Only applies to __new__ bypass — if __init__ ran, all attrs
            # should exist. First AttributeError on bypass may be an
            # __init__-time cache attr the fix added.
            attr_match = re.search(r"has no attribute '(\w+)'", str(e))
            if attr_match:
                missing = attr_match.group(1)
                # Narrow scope: only recover for attrs that look like
                # cache/format attrs AND are NOT standard nn.Module or
                # known required layer attrs.
                known_required = {
                    'nsa_enable_prefill_cp', 'self_attn', 'mlp',
                    'layer_communicator', 'layer_scatter_modes',
                    'config', 'hidden_size', 'training', 'forward',
                    'speculative_algorithm', 'is_layer_sparse',
                    'layer_id', 'is_nextn', 'input_layernorm',
                    'post_attention_layernorm',
                }
                if missing not in known_required and not missing.startswith('_'):
                    # Likely a cache attribute added by the fix
                    setattr(layer, missing, "mxfp4")
                    init_cache_evidence = True
                    continue  # retry this iteration
            break  # Unrecoverable AttributeError, stop
        else:
            break  # Not recoverable, stop
    except Exception:
        break

# ---- Analyze results ----
dtype_accesses = weight_proxy.access_count
quant_formats = quant_format_log

# Require non-empty + stable quant_formats (proves detection path was exercised
# and the same format is returned every time, without prescribing a specific value)
format_ok = bool(quant_formats) and len(set(quant_formats)) == 1

print(f"FWD_OK:{fwd_ok}")
print(f"DTYPE_ACCESSES:{dtype_accesses}")
print(f"QUANT_FORMATS_SAMPLE:{quant_formats[:5]}")
print(f"FORMAT_OK:{format_ok}")
print(f"INIT_CACHE_EVIDENCE:{init_cache_evidence}")

# PASS conditions (all three required):
# (a) forward() completed at least N/2 times
# (b) dtype access count is amortized (<= 1 across all calls)
# (c) quant_format values are non-empty and stable (same value every call)
fwd_sufficient = fwd_ok >= N // 2
dtype_amortized = dtype_accesses <= 1

amortized_ok = fwd_sufficient and dtype_amortized and format_ok

if amortized_ok:
    print("AMORTIZED_OK")
else:
    reasons = []
    if not fwd_sufficient:
        reasons.append(f"only {fwd_ok}/{N} forward() calls succeeded")
    if not dtype_amortized:
        reasons.append(
            f"weight.dtype accessed {dtype_accesses} times across {N} "
            f"forwards (expected <=1)"
        )
    if not format_ok:
        if not quant_formats:
            reasons.append("detection path not exercised (no quant_format captured)")
        else:
            reasons.append(f"quant_format unstable or wrong: {quant_formats[:3]}")
    print(f"AMORTIZED_FAIL:{'; '.join(reasons)}")
''')

result2 = subprocess.run(
    [sys.executable, "-c", amort_script],
    capture_output=True, text=True, timeout=120,
    cwd="/workspace",
    env={**os.environ, "PYTHONPATH": "/sgl-workspace/aiter"},
)

stdout2 = result2.stdout
stderr2 = result2.stderr

if "IMPORT_FAIL" in stdout2:
    detail = stdout2.split("IMPORT_FAIL:")[1].strip().split("\n")[0][:200]
    check("Quant format detection is amortized (dtype-access proof)", False,
          f"Import failed: {detail}")
elif "CONSTRUCT_FAIL" in stdout2:
    detail = stdout2.split("CONSTRUCT_FAIL:")[1].strip().split("\n")[0][:200]
    check("Quant format detection is amortized (dtype-access proof)", False,
          f"Layer construction failed: {detail}")
elif result2.returncode != 0 and "AMORTIZED" not in stdout2:
    check("Quant format detection is amortized (dtype-access proof)", False,
          f"Test error: {stderr2[:200]}")
else:
    passed = "AMORTIZED_OK" in stdout2
    detail = ""
    if not passed:
        for line in stdout2.strip().split("\n"):
            if "AMORTIZED_FAIL:" in line:
                detail = line.split("AMORTIZED_FAIL:")[1].strip()[:200]
                break
    check(
        "Quant format detection is amortized (dtype-access proof)",
        passed,
        detail if detail else "No amortization signal detected",
    )

# -------------------------------------------------------------------
# Check 3: Module structure intact
# -------------------------------------------------------------------
result3 = subprocess.run(
    [sys.executable, "-c", textwrap.dedent(r'''
import inspect
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

assert hasattr(DeepseekV2DecoderLayer, '__init__'), "missing __init__"
assert hasattr(DeepseekV2DecoderLayer, 'forward'), "missing forward"

sig = inspect.signature(DeepseekV2DecoderLayer.forward)
params = list(sig.parameters.keys())
assert len(params) >= 2, f"forward() has too few params: {params}"
print("STRUCTURE_OK")
''')],
    capture_output=True, text=True, timeout=30,
    cwd="/workspace",
)

check(
    "Module structure intact",
    "STRUCTURE_OK" in result3.stdout,
    result3.stderr[:200] if result3.returncode != 0 else "structure check failed",
)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0)
