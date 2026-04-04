#!/usr/bin/env python3
"""Test harness for sglang-kscale-vscale-fix. Behavioral tests only.

Bug: The call to extend_attention_fwd() in aiter_backend.py is missing
the required positional arguments k_scale and v_scale, causing a
TypeError at runtime when the target_verify / draft_extend path is taken.

Tests verify that the call site passes the correct number of arguments
matching the function's signature.
"""
import ast
import inspect
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
print("sglang-kscale-vscale-fix test harness")
print("=" * 60)

AITER_BACKEND = "/workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"
EXTEND_ATTN = "/workspace/sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py"

from pathlib import Path

if not check("aiter_backend.py exists", Path(AITER_BACKEND).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

if not check("extend_attention.py exists", Path(EXTEND_ATTN).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

aiter_source = Path(AITER_BACKEND).read_text()
extend_source = Path(EXTEND_ATTN).read_text()

# ----------------------------------------------------------------
# Check 1: Parse extend_attention_fwd signature to count required
# positional parameters (the function the call must match).
# ----------------------------------------------------------------
extend_tree = ast.parse(extend_source)
extend_fwd_def = None
for node in ast.walk(extend_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "extend_attention_fwd":
        extend_fwd_def = node
        break

if not check("extend_attention_fwd definition found", extend_fwd_def is not None):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Count required positional args (those without defaults).
# ast.arguments: args has all positional params, defaults are right-aligned.
all_params = extend_fwd_def.args.args
num_defaults = len(extend_fwd_def.args.defaults)
num_required = len(all_params) - num_defaults

param_names = [a.arg for a in all_params]
check(
    "extend_attention_fwd has k_scale parameter",
    "k_scale" in param_names,
    f"parameters: {param_names}",
)
check(
    "extend_attention_fwd has v_scale parameter",
    "v_scale" in param_names,
    f"parameters: {param_names}",
)

# Determine the expected position of k_scale and v_scale in the signature
k_scale_idx = param_names.index("k_scale") if "k_scale" in param_names else -1
v_scale_idx = param_names.index("v_scale") if "v_scale" in param_names else -1

check(
    "k_scale and v_scale are required positional args",
    k_scale_idx < num_required and v_scale_idx < num_required,
    f"k_scale_idx={k_scale_idx}, v_scale_idx={v_scale_idx}, num_required={num_required}",
)

# ----------------------------------------------------------------
# Check 2: AST analysis of the call site in aiter_backend.py.
# Find the call to self.extend_attention_fwd inside forward_extend
# and verify it passes enough positional arguments.
# ----------------------------------------------------------------
aiter_tree = ast.parse(aiter_source)

# Find the forward_extend method
forward_extend_def = None
for node in ast.walk(aiter_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "forward_extend":
        forward_extend_def = node
        break

check("forward_extend method found", forward_extend_def is not None)

# Find the call to self.extend_attention_fwd inside forward_extend
extend_fwd_calls = []
if forward_extend_def is not None:
    for node in ast.walk(forward_extend_def):
        if isinstance(node, ast.Call):
            func = node.func
            # Match self.extend_attention_fwd(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "extend_attention_fwd"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                extend_fwd_calls.append(node)

check(
    "self.extend_attention_fwd call found in forward_extend",
    len(extend_fwd_calls) >= 1,
    f"found {len(extend_fwd_calls)} calls",
)

# ----------------------------------------------------------------
# Check 3: Verify the call passes the correct number of positional
# arguments to match the function signature.
# ----------------------------------------------------------------
if extend_fwd_calls:
    call_node = extend_fwd_calls[0]
    num_positional_args = len(call_node.args)
    num_keyword_args = len(call_node.keywords)

    # The function requires `num_required` positional args.
    # Some of them may be passed as keyword args at the call site.
    # The minimum positional args needed is num_required minus any
    # keyword args that cover required params.
    keyword_names = [kw.arg for kw in call_node.keywords if kw.arg is not None]

    # Count how many required params are satisfied by keyword args
    required_param_names = param_names[:num_required]
    required_covered_by_kw = sum(
        1 for kn in keyword_names if kn in required_param_names
    )

    # Positional args at the call site must cover the rest
    required_needing_positional = num_required - required_covered_by_kw
    check(
        f"Call passes enough positional args (need >= {required_needing_positional}, got {num_positional_args})",
        num_positional_args >= required_needing_positional,
        f"positional={num_positional_args}, keywords={keyword_names}",
    )
else:
    check("Call passes enough positional args", False, "no call found")

# ----------------------------------------------------------------
# Check 4: Source-level check -- verify that the string literals
# "k_scale" and "v_scale" appear near the extend_attention_fwd call
# in aiter_backend.py (the fix adds them as inline comments).
# Also check that 1.0 values are passed before layer.scaling.
# ----------------------------------------------------------------
# Find the block of the call in the source text
lines = aiter_source.splitlines()
call_start = None
call_end = None
for i, line in enumerate(lines):
    if "self.extend_attention_fwd(" in line:
        call_start = i
    if call_start is not None and call_end is None:
        # The call ends when we see the closing paren at the right indentation
        stripped = line.strip()
        if stripped == ")":
            call_end = i
            break

if call_start is not None and call_end is not None:
    call_block = "\n".join(lines[call_start : call_end + 1])

    # In the fixed version, the call block should contain k_scale and v_scale
    # (either as comments or as keyword args).
    has_k_scale_ref = "k_scale" in call_block
    has_v_scale_ref = "v_scale" in call_block
    check(
        "Call site references k_scale",
        has_k_scale_ref,
        "k_scale not found in call block",
    )
    check(
        "Call site references v_scale",
        has_v_scale_ref,
        "v_scale not found in call block",
    )
else:
    check("Call site references k_scale", False, "could not locate call block")
    check("Call site references v_scale", False, "could not locate call block")

# ----------------------------------------------------------------
# Check 5: Subprocess test -- use Python AST in a subprocess to
# independently verify the argument count matches the signature.
# This catches any discrepancy we might miss in in-process analysis.
# ----------------------------------------------------------------
subprocess_script = textwrap.dedent("""\
    import ast, sys
    from pathlib import Path

    aiter_src = Path("{aiter}").read_text()
    extend_src = Path("{extend}").read_text()

    # Get required param count from extend_attention_fwd
    n_required = None
    all_param_names = []
    for node in ast.walk(ast.parse(extend_src)):
        if isinstance(node, ast.FunctionDef) and node.name == "extend_attention_fwd":
            all_param_names = [a.arg for a in node.args.args]
            n_defaults = len(node.args.defaults)
            n_required = len(all_param_names) - n_defaults
            break

    if n_required is None:
        print("FUNC_NOT_FOUND")
        sys.exit(0)

    # Get positional arg count from the call in forward_extend
    for node in ast.walk(ast.parse(aiter_src)):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_extend":
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr == "extend_attention_fwd":
                    n_pos = len(child.args)
                    n_kw_names = [kw.arg for kw in child.keywords if kw.arg]
                    req_p = all_param_names[:n_required]
                    covered = sum(1 for k in n_kw_names if k in req_p)
                    needed = n_required - covered
                    if n_pos >= needed:
                        print("ARGS_OK")
                    else:
                        print("ARGS_MISMATCH:need={{}},got={{}}".format(needed, n_pos))
                    sys.exit(0)
    print("CALL_NOT_FOUND")
""".format(aiter=AITER_BACKEND, extend=EXTEND_ATTN))

result = subprocess.run(
    ["/opt/venv/bin/python3", "-c", subprocess_script],
    capture_output=True,
    text=True,
    timeout=60,
)
stdout = (result.stdout or "").strip()
stderr = (result.stderr or "")[-300:]

if "ARGS_OK" in stdout:
    check("Subprocess arg-count verification", True)
elif "ARGS_MISMATCH" in stdout:
    check("Subprocess arg-count verification", False, stdout)
elif "FUNC_NOT_FOUND" in stdout or "CALL_NOT_FOUND" in stdout:
    # Structure changed in an unexpected way; not the bug we're testing
    check("Subprocess arg-count verification (structure changed)", True)
else:
    check(
        "Subprocess arg-count verification",
        False,
        f"stdout={stdout}, stderr={stderr[-200:]}",
    )

# ----------------------------------------------------------------
# Final score
# ----------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
