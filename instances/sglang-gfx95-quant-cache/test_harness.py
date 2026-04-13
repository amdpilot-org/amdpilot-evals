#!/usr/bin/env python3
"""Test harness for sglang-gfx95-quant-cache.

Verifies that quantization format detection in the decoder layer is amortized
(performed at most once) rather than on every forward() call.  Detection
amortization is verified through runtime instance-state inspection and
property/decorator checks on the constructed layer — no source code analysis.
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
    check("Quant format detection is amortized", False,
          "import failed")
    check("Module structure intact", False,
          "import failed")
    print()
    print(f"Results: {checks_passed}/{checks_total}")
    print(f"SCORE: 0.0")
    sys.exit(0)

check("Import decoder layer class", True)

# -------------------------------------------------------------------
# Check 2: Detection amortization — runtime behavioral check
#
# Constructs a DeepseekV2DecoderLayer on the meta device and inspects
# its instance state for evidence that quant-format detection results
# are stored at construction time (or lazily via a cached property /
# lru_cache), rather than computed inside forward() on every call.
#
# Pre-fix behaviour: forward() inspects weight dtypes inline every
# call — no cached attribute exists after __init__.
# Post-fix behaviour: __init__ (or a one-shot helper) stores the
# detected format as an instance attribute or cached property.
# -------------------------------------------------------------------
amort_script = textwrap.dedent(r'''
import json, os, sys, tempfile, types
import torch

from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

# ---- build a minimal config ----
try:
    from transformers import AutoConfig
except ImportError:
    print("CONFIG_FAIL:transformers not available")
    sys.exit(1)

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
}

tf = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, dir="/tmp"
)
json.dump(min_cfg, tf)
tf.close()
try:
    config = AutoConfig.from_pretrained(tf.name, trust_remote_code=True)
except Exception as e:
    print(f"CONFIG_FAIL:{e}")
    sys.exit(1)
finally:
    os.unlink(tf.name)

# ---- construct the layer on the meta device ----
try:
    with torch.device("meta"):
        layer = DeepseekV2DecoderLayer(config, layer_idx=0)
except TypeError:
    try:
        with torch.device("meta"):
            layer = DeepseekV2DecoderLayer(config, 0)
    except Exception as e:
        print(f"CONSTRUCT_FAIL:{e}")
        sys.exit(1)
except Exception as e:
    print(f"CONSTRUCT_FAIL:{e}")
    sys.exit(1)

# ---- look for cached quant-format state ----
has_cache = False
cache_attr = None

# Strategy 1: instance attribute whose NAME indicates cached quant format.
# (Includes None-valued attrs — their existence proves __init__ set them up.)
for k in layer.__dict__:
    k_lower = k.lower()
    if any(
        kw in k_lower
        for kw in [
            "quant_format", "quant_cache", "cached_quant", "_quant_fmt",
            "quant_type", "weight_format", "gfx_quant", "w_quant",
            "_gfx95", "_gfx950", "quantization_format",
        ]
    ):
        has_cache = True
        cache_attr = k
        break

# Strategy 2: instance attribute whose VALUE is a quant-format string.
if not has_cache:
    for k, v in layer.__dict__.items():
        if isinstance(v, (torch.Tensor, torch.nn.Module)):
            continue
        if isinstance(v, str):
            v_lower = v.lower()
            if any(
                fmt in v_lower
                for fmt in ["fp8", "int8", "uint8", "float8", "mxfp"]
            ):
                has_cache = True
                cache_attr = k
                break

# Strategy 3: cached_property or lru_cache on the class.
if not has_cache:
    for attr_name in dir(type(layer)):
        obj = getattr(type(layer), attr_name, None)
        if obj is None:
            continue
        if type(obj).__name__ == "cached_property":
            attr_lower = attr_name.lower()
            if any(kw in attr_lower for kw in ["quant", "format", "dtype", "gfx"]):
                has_cache = True
                cache_attr = attr_name
                break
        if hasattr(obj, "cache_info"):
            has_cache = True
            cache_attr = attr_name
            break

print(f"HAS_CACHE:{has_cache}")
print(f"CACHE_ATTR:{cache_attr}")
''')

result2 = subprocess.run(
    [sys.executable, "-c", amort_script],
    capture_output=True, text=True, timeout=120,
    cwd="/workspace",
    env={**os.environ, "PYTHONPATH": "/sgl-workspace/aiter"},
)

stdout2 = result2.stdout
stderr2 = result2.stderr

if "CONSTRUCT_FAIL" in stdout2:
    detail = stdout2.split("CONSTRUCT_FAIL:")[1].strip().split("\n")[0][:200]
    check("Quant format detection is amortized", False,
          f"Layer construction failed: {detail}")
elif "CONFIG_FAIL" in stdout2:
    detail = stdout2.split("CONFIG_FAIL:")[1].strip().split("\n")[0][:200]
    check("Quant format detection is amortized", False,
          f"Config creation failed: {detail}")
elif result2.returncode != 0:
    check("Quant format detection is amortized", False,
          f"Test error: {stderr2[:200]}")
else:
    cached = "HAS_CACHE:True" in stdout2
    check(
        "Quant format detection is amortized",
        cached,
        "No cached quant format found in instance state after "
        "construction — detection may still run per-forward",
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
