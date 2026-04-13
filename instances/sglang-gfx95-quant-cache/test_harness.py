#!/usr/bin/env python3
"""Test harness for sglang-gfx95-quant-cache.

Verifies that DeepseekV2DecoderLayer caches gfx95 quant format detection
instead of recomputing it on every forward() call.
"""
import ast
import re
import sys

sys.path.insert(0, "/workspace/sglang/python")

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

# Read the source file
src_path = "/workspace/sglang/python/sglang/srt/models/deepseek_v2.py"
try:
    with open(src_path) as f:
        source = f.read()
    tree = ast.parse(source)
    check("Parse deepseek_v2.py", True)
except Exception as e:
    check("Parse deepseek_v2.py", False, str(e)[:200])
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Find the DeepseekV2DecoderLayer class
decoder_layer = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2DecoderLayer":
        decoder_layer = node
        break

if not check("Find DeepseekV2DecoderLayer class", decoder_layer is not None):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Find forward() method
forward_method = None
for node in ast.iter_child_nodes(decoder_layer):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name == "forward":
            forward_method = node
            break

if not check("Find forward() method", forward_method is not None):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

forward_lines = source.split("\n")[forward_method.lineno - 1:forward_method.end_lineno]
forward_src = "\n".join(forward_lines)

# Check 1: forward() should NOT contain inline dtype detection
has_dtype_check = (
    "torch.uint8" in forward_src or
    "torch.float8_e4m3fn" in forward_src or
    "float8_e4m3fn" in forward_src
)
has_getattr_chain = "fused_qkv_a_proj_with_mqa" in forward_src

check(
    "No inline dtype detection in forward()",
    not has_dtype_check,
    "forward() still contains dtype checks — should be cached"
)

check(
    "No getattr weight inspection in forward()",
    not has_getattr_chain,
    "forward() still inspects weight tensors — should be cached"
)

# Check 2: __init__ or a helper should cache the quant format
init_method = None
for node in ast.iter_child_nodes(decoder_layer):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name == "__init__":
            init_method = node
            break

if not check("Find __init__() method", init_method is not None):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

init_lines = source.split("\n")[init_method.lineno - 1:init_method.end_lineno]
init_src = "\n".join(init_lines)

has_quant_cache = bool(re.search(r"self\.\w*quant\w*", init_src))

check(
    "Quant format cached in __init__",
    has_quant_cache,
    "__init__() does not set up quant format caching"
)

# Check 3: forward() should reference the cached attribute
has_self_quant_ref = bool(re.search(r"self\.\w*quant\w*", forward_src))
check(
    "forward() uses cached quant format",
    has_self_quant_ref,
    "forward() does not reference a cached self.* quant attribute"
)

# Check 4: Module imports successfully
try:
    from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
    check("Import DeepseekV2DecoderLayer", True)
except Exception as e:
    check("Import DeepseekV2DecoderLayer", False, str(e)[:200])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.2f}")
sys.exit(0 if checks_passed == checks_total else 1)
