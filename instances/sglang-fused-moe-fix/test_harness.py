#!/usr/bin/env python3
"""Test harness for sglang-fused-moe-fix. Behavioral tests only.

Bug: MoE models crash on ROCm with NameError in fused_moe.py.
     get_global_server_args is imported only inside `if _is_cuda:` but used
     unconditionally in fused_experts_impl(). On HIP, calling fused_experts_impl()
     raises NameError.

Test: Verify the module imports correctly AND that names used at runtime
      (specifically get_global_server_args) are accessible on HIP.
"""
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
print("sglang-fused-moe-fix test harness")
print("=" * 60)

TARGET = "/workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py"

from pathlib import Path
if not check("Target file exists", Path(TARGET).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source = Path(TARGET).read_text()

# Check 1: File is valid Python
try:
    import ast
    tree = ast.parse(source)
    check("Valid Python", True)
except SyntaxError as e:
    check("Valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 2: Module imports without NameError on ROCm
# NOTE: Import alone won't trigger the bug because get_global_server_args
# is only called at runtime inside fused_experts_impl(), not at import time.
fused_moe_module = None
try:
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module
    check("Module imports without NameError", True)
except NameError as e:
    check("Module imports without NameError", False, f"NameError: {e}")
except (ImportError, ModuleNotFoundError) as e:
    # Import failures due to missing dependencies (sgl_kernel, triton, etc.)
    # are not the bug we're testing for — still need to check source-level.
    check("Module imports (dep issue, not target bug)", True)
except Exception as e:
    # Other exceptions during import are also not the NameError bug.
    # But we should NOT blindly pass — log it.
    check("Module imports (other exception)", False, f"{type(e).__name__}: {str(e)[:200]}")

# Check 3: get_global_server_args is accessible on HIP path
# This is the CORE check. The bug is that get_global_server_args is only
# imported in the `if _is_cuda:` branch, so on HIP it's undefined when
# fused_experts_impl() tries to call it.
#
# We check this via AST analysis: if get_global_server_args is used in a
# function body but only imported conditionally under _is_cuda, that's the bug.
import subprocess
result = subprocess.run(
    ["/opt/venv/bin/python3", "-c", """
import sys
sys.path.insert(0, '/workspace/sglang/python')
import ast
from pathlib import Path

source = Path('/workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py').read_text()
tree = ast.parse(source)

# Find all names used in function bodies (runtime calls)
runtime_names = set()
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                runtime_names.add(child.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                runtime_names.add(child.func.id)

# Find all top-level imports (unconditional)
unconditional_imports = set()
for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            unconditional_imports.add(alias.asname if alias.asname else alias.name)

# Find conditional imports (inside If blocks at module level)
conditional_imports = {}  # name -> condition description
for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.If):
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    name = alias.asname if alias.asname else alias.name
                    # Try to get condition text
                    try:
                        cond = ast.dump(node.test)
                    except:
                        cond = "unknown"
                    conditional_imports[name] = cond

# Check if get_global_server_args is used at runtime but only conditionally imported
target_name = 'get_global_server_args'
if target_name in runtime_names:
    if target_name in unconditional_imports:
        print('ACCESSIBLE')  # imported unconditionally, OK
    elif target_name in conditional_imports:
        print('CONDITIONAL_ONLY')  # only in conditional import -> BUG on other platforms
    else:
        # Check if it's defined as a function/variable at module level
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.Assign)):
                if isinstance(node, ast.FunctionDef) and node.name == target_name:
                    print('ACCESSIBLE')
                    break
        else:
            print('MISSING')  # not imported or defined at all
else:
    print('NOT_USED')  # not used at runtime, so no bug
"""],
    capture_output=True, text=True, timeout=30,
)
ast_output = (result.stdout or "").strip()
ast_err = (result.stderr or "").strip()

if ast_output == "ACCESSIBLE" or ast_output == "NOT_USED":
    check("get_global_server_args accessible on all platforms", True)
elif ast_output == "CONDITIONAL_ONLY":
    check("get_global_server_args accessible on all platforms", False,
          "Only imported conditionally (e.g. under _is_cuda). Will NameError on HIP.")
elif ast_output == "MISSING":
    check("get_global_server_args accessible on all platforms", False,
          "Not imported or defined at module level. Will NameError at runtime.")
else:
    check("get_global_server_args accessible on all platforms", False,
          f"AST check failed: stdout={ast_output!r}, stderr={ast_err[:200]}")

# Check 4: Runtime verification in subprocess — actually try to reference
# get_global_server_args in the module namespace after import on HIP
result2 = subprocess.run(
    ["/opt/venv/bin/python3", "-c",
     "import sys; sys.path.insert(0, '/workspace/sglang/python'); "
     "from sglang.srt.layers.moe.fused_moe_triton import fused_moe; "
     "fn = getattr(fused_moe, 'get_global_server_args', 'MISSING'); "
     "print('MODULE_HAS_IT' if fn != 'MISSING' else 'MODULE_MISSING')"],
    capture_output=True, text=True, timeout=60,
)
rt_output = (result2.stdout or "").strip()
rt_err = (result2.stderr or "")[-300:]

if "MODULE_HAS_IT" in rt_output:
    check("get_global_server_args in module namespace at runtime", True)
elif "MODULE_MISSING" in rt_output:
    check("get_global_server_args in module namespace at runtime", False,
          "Not in module namespace — will crash when fused_experts_impl() is called")
elif "NameError" in rt_err:
    check("get_global_server_args in module namespace at runtime", False,
          f"NameError during import: {rt_err[-200:]}")
elif "ImportError" in rt_err or "ModuleNotFoundError" in rt_err:
    # Can't fully check at runtime due to missing deps, rely on AST check
    check("get_global_server_args in module namespace at runtime (dep issue, rely on AST)", True)
else:
    check("get_global_server_args in module namespace at runtime", False,
          f"Unexpected: stdout={rt_output[:100]}, stderr={rt_err[:200]}")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
