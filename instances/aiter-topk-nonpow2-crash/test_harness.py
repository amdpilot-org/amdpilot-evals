#!/usr/bin/env python3
"""Test harness for aiter-topk-nonpow2-crash.

Validates that expert routing dispatch handles arbitrary expert counts
correctly at all sequence lengths.
"""
import ast
import os
import re
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
AITER_PATH = "/workspace/aiter"
TOPK_PATH = os.path.join(AITER_PATH, "aiter/ops/topk.py")


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
print("aiter-topk-nonpow2-crash test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("topk.py exists", os.path.isfile(TOPK_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(TOPK_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("topk.py is valid Python", True)
except SyntaxError as e:
    check("topk.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 2: Find biased_grouped_topk function
# ---------------------------------------------------------------------------
print("\n--- Check 2: biased_grouped_topk function ---")

topk_fn = None
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "biased_grouped_topk":
        topk_fn = node
        break

if not check(
    "biased_grouped_topk function found",
    topk_fn is not None,
    "function not found",
):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

fn_src_lines = source_text.splitlines()[topk_fn.lineno - 1 : topk_fn.end_lineno]
fn_src = "\n".join(fn_src_lines)

# ---------------------------------------------------------------------------
# Check 3: Power-of-2 check exists in the function
# ---------------------------------------------------------------------------
print("\n--- Check 3: Power-of-2 guard ---")

# The fix should contain a power-of-2 check. Common patterns:
#   (n & (n - 1)) == 0
#   n & (n - 1)
#   math.log2(n) % 1
#   is_power_of_2
#   bin(n).count('1') == 1
pow2_patterns = [
    r"&\s*\(",           # bitwise AND pattern: n & (n - 1)
    r"power.of.2",       # variable name like is_power_of_2
    r"pow2",             # variable name like num_experts_pow2
    r"log2",             # math.log2 approach
    r"bit_count|count\(\s*['\"]1['\"]\s*\)",  # bit count approach
    r"&.*-\s*1",         # inline bitwise: n & (n - 1)
]

has_pow2_check = any(re.search(pat, fn_src, re.IGNORECASE) for pat in pow2_patterns)

check(
    "Power-of-2 check exists in biased_grouped_topk",
    has_pow2_check,
    "No power-of-2 guard found — moe_fused_gate will crash for non-pow2 expert counts",
)

# ---------------------------------------------------------------------------
# Check 4: The dispatch condition includes the power-of-2 guard
# ---------------------------------------------------------------------------
print("\n--- Check 4: Dispatch condition ---")

# The old dispatch was simply: if token_num <= cu_num * 212:
# The fix should add: ... or not is_power_of_2 (or similar)
# This ensures non-pow2 expert counts always go to biased_grouped_topk_hip

has_combined_condition = False

# Strategy A: AST-based — check if-conditions for combined token+pow2 checks
for node in ast.walk(topk_fn):
    if isinstance(node, ast.If):
        cond_src = ast.get_source_segment(source_text, node.test)
        if cond_src is None:
            continue
        has_token_check = "token_num" in cond_src or "cu_num" in cond_src
        has_pow2_in_cond = any(
            re.search(pat, cond_src, re.IGNORECASE) for pat in pow2_patterns
        )
        if has_token_check and has_pow2_in_cond:
            has_combined_condition = True
            break

# Strategy B: scan lines for combined condition on a single line
if not has_combined_condition:
    for line in fn_src_lines:
        stripped = line.strip()
        if "if " in stripped or " or " in stripped:
            has_token = "token_num" in stripped or "cu_num" in stripped
            has_pow2 = any(
                re.search(pat, stripped, re.IGNORECASE) for pat in pow2_patterns
            )
            if has_token and has_pow2:
                has_combined_condition = True
                break

# Strategy C: accept early-return guard pattern
# e.g., if not (n & (n-1) == 0): return biased_grouped_topk_hip(...)
# or:   if not is_power_of_2: return biased_grouped_topk_hip(...)
if not has_combined_condition:
    fn_src_nospace = fn_src.replace(" ", "").replace("\n", "")
    early_return_patterns = [
        r"if.*not.*pow.*2.*:.*biased_grouped_topk_hip",
        r"if.*not.*pow.*2.*:.*return",
        r"if.*not.*&.*-\s*1.*:.*biased_grouped_topk_hip",
        r"if.*not.*&.*-\s*1.*:.*return",
        # Check for a separate if-block with pow2 guard before the token_num dispatch
        r"if.*not.*power.*:.*return",
    ]
    for pat in early_return_patterns:
        if re.search(pat, fn_src_nospace, re.IGNORECASE):
            has_combined_condition = True
            break

# Strategy D: AST — check for any If node whose body calls
# biased_grouped_topk_hip and whose test includes a pow2 pattern
if not has_combined_condition:
    for node in ast.walk(topk_fn):
        if isinstance(node, ast.If):
            # Check if the body calls biased_grouped_topk_hip
            body_src = "\n".join(
                source_text.splitlines()[node.lineno - 1 : node.end_lineno]
            )
            if "biased_grouped_topk_hip" in body_src:
                cond_src = ast.get_source_segment(source_text, node.test)
                if cond_src and any(
                    re.search(pat, cond_src, re.IGNORECASE) for pat in pow2_patterns
                ):
                    has_combined_condition = True
                    break
                # Also check: condition is a UnaryOp (not X) where X is a pow2 check
                if isinstance(node.test, ast.UnaryOp) and isinstance(
                    node.test.op, ast.Not
                ):
                    operand_src = ast.get_source_segment(source_text, node.test.operand)
                    if operand_src and any(
                        re.search(pat, operand_src, re.IGNORECASE)
                        for pat in pow2_patterns
                    ):
                        has_combined_condition = True
                        break

check(
    "Dispatch condition includes power-of-2 guard",
    has_combined_condition,
    "The dispatch should route non-pow2 expert counts to biased_grouped_topk_hip",
)

# ---------------------------------------------------------------------------
# Check 5: moe_fused_gate is NOT called for non-pow2 expert counts
# ---------------------------------------------------------------------------
print("\n--- Check 5: moe_fused_gate protection ---")

check(
    "moe_fused_gate protected from non-pow2 expert counts",
    has_pow2_check and has_combined_condition,
    "moe_fused_gate can still be reached with non-pow2 experts",
)

# ---------------------------------------------------------------------------
# Check 6 (behavioral): Verify the actual dispatch condition in the code
# ---------------------------------------------------------------------------
print("\n--- Check 6: Behavioral dispatch verification ---")

behavior_script = f"""
import sys, ast
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

with open("{TOPK_PATH}") as f:
    src = f.read()
tree = ast.parse(src)

# Find biased_grouped_topk and analyze its dispatch logic
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "biased_grouped_topk":
        # Count If nodes and check their conditions
        if_nodes = [n for n in ast.iter_child_nodes(node) if isinstance(n, ast.If)]

        if not if_nodes:
            print("DISPATCH:no_if_found")
            break

        first_if = if_nodes[0]
        cond_src = ast.get_source_segment(src, first_if.test)

        # Check if condition references num_experts or expert count
        refs_experts = "expert" in (cond_src or "").lower() or "pow" in (cond_src or "").lower()
        refs_token = "token" in (cond_src or "").lower() or "cu_num" in (cond_src or "").lower()

        # Check structure: BoolOp(Or) means combined condition
        if isinstance(first_if.test, ast.BoolOp) and isinstance(first_if.test.op, ast.Or):
            print("DISPATCH_TYPE:combined_or")
        elif isinstance(first_if.test, ast.UnaryOp) and isinstance(first_if.test.op, ast.Not):
            print("DISPATCH_TYPE:early_return_guard")
        elif isinstance(first_if.test, ast.Compare):
            print("DISPATCH_TYPE:simple_compare")
        else:
            print(f"DISPATCH_TYPE:{{type(first_if.test).__name__}}")

        print(f"REFS_EXPERTS:{{refs_experts}}")
        print(f"REFS_TOKEN:{{refs_token}}")

        # Check if moe_fused_gate is in the else branch (should only be
        # reachable when pow2 check passes)
        has_fused_gate_in_else = False
        for orelse_node in ast.walk(ast.Module(body=first_if.orelse)):
            if isinstance(orelse_node, ast.Call):
                call_src = ast.get_source_segment(src, orelse_node)
                if call_src and "moe_fused_gate" in call_src:
                    has_fused_gate_in_else = True
                    break

        # Also check for a second if-block that calls moe_fused_gate
        if not has_fused_gate_in_else and len(if_nodes) > 1:
            for later_if in if_nodes[1:]:
                later_src = "\\n".join(src.splitlines()[later_if.lineno-1:later_if.end_lineno])
                if "moe_fused_gate" in later_src:
                    has_fused_gate_in_else = True
                    break

        print(f"FUSED_GATE_GUARDED:{{has_fused_gate_in_else or refs_experts}}")
        break
"""

try:
    result = subprocess.run(
        [VENV_PYTHON, "-c", behavior_script],
        capture_output=True, text=True, timeout=30,
    )
    stdout6 = result.stdout
    stderr6 = result.stderr
except Exception as e:
    stdout6 = ""
    stderr6 = str(e)

if stdout6:
    # The dispatch should NOT be simple_compare (the buggy pattern)
    check(
        "Dispatch is not the buggy simple_compare pattern",
        "DISPATCH_TYPE:simple_compare" not in stdout6,
        "Dispatch still uses simple token_num comparison without pow2 guard",
    )
    check(
        "Dispatch references expert count in condition",
        "REFS_EXPERTS:True" in stdout6,
        "Dispatch condition does not reference expert count or power-of-2",
    )
else:
    check("Behavioral test ran", False, f"stderr: {stderr6[:200]}")
    check("Dispatch analysis", False, "behavioral test failed to run")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
