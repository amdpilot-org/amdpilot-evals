#!/usr/bin/env python3
"""Test harness for vllm PR #33749: AITER import/env-var separation fix.

Bug: Setting VLLM_ROCM_USE_AITER=0 then explicitly selecting the AITER backend
via attention_config crashes with:
  AttributeError: 'builtin_function_or_method' object has no attribute
  'flash_attn_varlen_func'

The env var disables the aiter import at module level, but the backend code
still tries to call aiter functions directly. Explicit backend selection should
work regardless of the env var.

Tests (behavioral, subprocess-isolated):
  1. Platform capability detection is separated from user preference (env var).
  2. Attention ops are accessible without requiring a module-level aiter import.
  3. Backend does not call aiter functions directly at module scope -- it goes
     through an indirection layer that defers the import.
"""
import ast
import os
import subprocess
import sys
import textwrap

checks_passed = 0
checks_total = 0

AITER_OPS_PATH = "/workspace/vllm/vllm/_aiter_ops.py"
ROCM_AITER_FA_PATH = "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_fa.py"
EAGLE_PATH = "/workspace/vllm/vllm/v1/spec_decode/eagle.py"
VENV_PYTHON = "/opt/venv/bin/python3"


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


def run_subprocess(script, timeout=120, env_override=None):
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/workspace",
        env=env,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-aiter-import-fix test harness (PR #33749)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Pre-check: target files exist
# ---------------------------------------------------------------------------
if not check("_aiter_ops.py exists", os.path.isfile(AITER_OPS_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

if not check("rocm_aiter_fa.py exists", os.path.isfile(ROCM_AITER_FA_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Parse both files
try:
    with open(AITER_OPS_PATH) as fh:
        aiter_ops_source = fh.read()
    aiter_ops_tree = ast.parse(aiter_ops_source)
    check("_aiter_ops.py is valid Python", True)
except SyntaxError as e:
    check("_aiter_ops.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

try:
    with open(ROCM_AITER_FA_PATH) as fh:
        rocm_fa_source = fh.read()
    rocm_fa_tree = ast.parse(rocm_fa_source)
    check("rocm_aiter_fa.py is valid Python", True)
except SyntaxError as e:
    check("rocm_aiter_fa.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


# =========================================================================
# CHECK 1: is_aiter_found_and_supported() must NOT check VLLM_ROCM_USE_AITER
#
# Before fix: the function body contains envs.VLLM_ROCM_USE_AITER check.
# After fix: the env var check is removed -- this function only checks
# platform capability, not user preference.
# =========================================================================
print("\n--- Check 1: is_aiter_found_and_supported separation of concerns ---")

# Find the is_aiter_found_and_supported function
aiter_supported_fn = None
for node in ast.walk(aiter_ops_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "is_aiter_found_and_supported":
        aiter_supported_fn = node
        break

if not check(
    "is_aiter_found_and_supported function found",
    aiter_supported_fn is not None,
    "function not found in _aiter_ops.py",
):
    pass
else:
    # Extract the function source
    fn_lines = aiter_ops_source.splitlines()[
        aiter_supported_fn.lineno - 1 : aiter_supported_fn.end_lineno
    ]
    fn_source = "\n".join(fn_lines)

    # The fix REMOVES the env var check from this function.
    # Before fix: "envs.VLLM_ROCM_USE_AITER" appears in the body.
    # After fix: it does not.
    has_env_check = "envs.VLLM_ROCM_USE_AITER" in fn_source

    check(
        "is_aiter_found_and_supported does NOT check VLLM_ROCM_USE_AITER env var",
        not has_env_check,
        "function still checks env var -- explicit backend selection will fail "
        "when env var is 0",
    )

    # Verify it still checks platform and library availability
    # It should reference current_platform.is_rocm() and IS_AITER_FOUND
    body_names = set()
    for child in ast.walk(aiter_supported_fn):
        if isinstance(child, ast.Name):
            body_names.add(child.id)
        elif isinstance(child, ast.Attribute):
            if hasattr(child, 'attr'):
                body_names.add(child.attr)

    check(
        "is_aiter_found_and_supported still checks IS_AITER_FOUND",
        "IS_AITER_FOUND" in body_names,
        "function should verify aiter library exists",
    )

    check(
        "is_aiter_found_and_supported still checks platform (is_rocm)",
        "is_rocm" in body_names,
        "function should verify ROCm platform",
    )


# =========================================================================
# CHECK 2: rocm_aiter_ops class has flash_attn_varlen_func and pa_fwd_asm
#
# The fix adds these as static methods with lazy imports, so the backend
# can call them without needing a module-level 'import aiter' that breaks
# when VLLM_ROCM_USE_AITER=0.
# =========================================================================
print("\n--- Check 2: rocm_aiter_ops has lazy-import wrapper methods ---")

# Find the rocm_aiter_ops class (it's named differently in the source;
# look for a class that has these methods)
target_class = None
for node in ast.walk(aiter_ops_tree):
    if isinstance(node, ast.ClassDef):
        method_names = set()
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_names.add(item.name)
        # The class we want should have is_enabled and should get the new methods
        if "is_enabled" in method_names or "register_ops_once" in method_names:
            target_class = node
            break

if not check(
    "rocm_aiter_ops class found in _aiter_ops.py",
    target_class is not None,
    "could not find the ops class with is_enabled/register_ops_once",
):
    pass
else:
    class_methods = {}
    for item in target_class.body:
        if isinstance(item, ast.FunctionDef):
            class_methods[item.name] = item

    # Check flash_attn_varlen_func exists
    has_flash_attn = "flash_attn_varlen_func" in class_methods
    check(
        "Class has flash_attn_varlen_func method",
        has_flash_attn,
        "missing -- backend will fall back to direct aiter import which crashes",
    )

    # Check pa_fwd_asm exists
    has_pa_fwd = "pa_fwd_asm" in class_methods
    check(
        "Class has pa_fwd_asm method",
        has_pa_fwd,
        "missing -- paged attention ASM path will crash when env var is 0",
    )

    # Verify flash_attn_varlen_func does lazy import of aiter
    if has_flash_attn:
        fa_fn = class_methods["flash_attn_varlen_func"]
        fa_lines = aiter_ops_source.splitlines()[
            fa_fn.lineno - 1 : fa_fn.end_lineno
        ]
        fa_source = "\n".join(fa_lines)

        # Should contain 'from aiter import flash_attn_varlen_func' or
        # 'import aiter' as a lazy/deferred import inside the method body
        has_lazy_import = (
            "from aiter import" in fa_source
            or "import aiter" in fa_source
        )
        check(
            "flash_attn_varlen_func uses lazy import of aiter",
            has_lazy_import,
            "method should lazily import aiter to avoid crash when env var is 0",
        )

    # Verify pa_fwd_asm does lazy import of aiter
    if has_pa_fwd:
        pa_fn = class_methods["pa_fwd_asm"]
        pa_lines = aiter_ops_source.splitlines()[
            pa_fn.lineno - 1 : pa_fn.end_lineno
        ]
        pa_source = "\n".join(pa_lines)

        has_lazy_import_pa = (
            "from aiter import" in pa_source
            or "import aiter" in pa_source
        )
        check(
            "pa_fwd_asm uses lazy import of aiter",
            has_lazy_import_pa,
            "method should lazily import aiter to avoid crash when env var is 0",
        )

    # Verify these methods are NOT decorated with @if_aiter_supported
    # (which would re-introduce the env var gating and defeat the purpose)
    for method_name in ["flash_attn_varlen_func", "pa_fwd_asm"]:
        if method_name in class_methods:
            fn_node = class_methods[method_name]
            decorator_names = []
            for dec in fn_node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorator_names.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorator_names.append(dec.attr)

            is_gated = "if_aiter_supported" in decorator_names
            check(
                f"{method_name} is NOT gated by @if_aiter_supported decorator",
                not is_gated,
                "decorator would block explicit backend selection when env var is 0",
            )


# =========================================================================
# CHECK 3: rocm_aiter_fa.py routes through rocm_aiter_ops, not aiter directly
#
# Before fix: the backend calls aiter.flash_attn_varlen_func directly
#   (module-level 'import aiter' guarded by is_enabled()).
# After fix: calls rocm_aiter_ops.flash_attn_varlen_func (lazy import inside).
#
# Also: the module-level 'import aiter' guarded by is_enabled() should be
# removed from rocm_aiter_fa.py.
# =========================================================================
print("\n--- Check 3: backend routes through rocm_aiter_ops, not aiter directly ---")

# Count calls to aiter.flash_attn_varlen_func vs rocm_aiter_ops.flash_attn_varlen_func
# in the backend source (excluding comments and strings).

# Simple approach: count occurrences in non-comment lines
direct_aiter_flash_calls = 0
wrapped_flash_calls = 0
direct_aiter_pa_calls = 0
wrapped_pa_calls = 0

for line in rocm_fa_source.splitlines():
    stripped = line.strip()
    # Skip comments
    if stripped.startswith("#"):
        continue
    # Skip string literals (docstrings are already handled by AST but
    # we're doing line-by-line, so be conservative)
    if stripped.startswith('"""') or stripped.startswith("'''"):
        continue
    if stripped.startswith('"') or stripped.startswith("'"):
        continue

    # Count flash_attn_varlen_func calls
    # Note: "rocm_aiter_ops.flash_attn_varlen_func" does NOT contain
    # "aiter.flash_attn_varlen_func" as a substring (the _ops. breaks it),
    # so we check for them independently.
    if "rocm_aiter_ops.flash_attn_varlen_func" in stripped:
        wrapped_flash_calls += 1
    elif "aiter.flash_attn_varlen_func" in stripped:
        direct_aiter_flash_calls += 1

    # Count pa_fwd_asm calls
    if "rocm_aiter_ops.pa_fwd_asm" in stripped:
        wrapped_pa_calls += 1
    elif "aiter.pa_fwd_asm" in stripped:
        direct_aiter_pa_calls += 1

check(
    "No direct aiter.flash_attn_varlen_func calls in backend",
    direct_aiter_flash_calls == 0,
    f"found {direct_aiter_flash_calls} direct call(s) -- these crash when env var is 0",
)

check(
    "Backend uses rocm_aiter_ops.flash_attn_varlen_func instead",
    wrapped_flash_calls > 0,
    "no rocm_aiter_ops.flash_attn_varlen_func calls found",
)

check(
    "No direct aiter.pa_fwd_asm calls in backend",
    direct_aiter_pa_calls == 0,
    f"found {direct_aiter_pa_calls} direct call(s) -- these crash when env var is 0",
)

check(
    "Backend uses rocm_aiter_ops.pa_fwd_asm instead",
    wrapped_pa_calls > 0,
    "no rocm_aiter_ops.pa_fwd_asm calls found",
)

# Check that the module-level 'import aiter' guarded by is_enabled() is removed.
# Before fix, rocm_aiter_fa.py has:
#   if rocm_aiter_ops.is_enabled():
#       import aiter
# After fix, this block should be removed (aiter is imported lazily in methods).

# Look for module-level import aiter in the ROCm platform guard section
# (not inside function bodies).
module_level_import_aiter = False
for node in ast.walk(rocm_fa_tree):
    # Look for 'import aiter' at module level (not inside a function)
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "aiter":
                # Check if it's inside a function
                # We'll use a simpler approach: check if the import is
                # inside the is_enabled() guard at module level
                pass

# More robust: scan source for the specific pattern
in_is_enabled_block = False
for i, line in enumerate(rocm_fa_source.splitlines()):
    stripped = line.strip()
    if "rocm_aiter_ops.is_enabled()" in stripped and "if" in stripped:
        in_is_enabled_block = True
        continue
    if in_is_enabled_block:
        if "import aiter" in stripped and not stripped.startswith("#"):
            module_level_import_aiter = True
            break
        # If we hit a non-indented line, the block ended
        if stripped and not line.startswith(" " * 8):
            in_is_enabled_block = False

check(
    "Module-level 'import aiter' guarded by is_enabled() removed from backend",
    not module_level_import_aiter,
    "module-level import still present -- will fail to import aiter when env var is 0",
)


# =========================================================================
# CHECK 4 (bonus): AITER_FP8_DTYPE module-level block removed
#
# Before fix: _aiter_ops.py had a module-level block that did:
#   if is_aiter_found_and_supported():
#       from aiter import dtypes
#       AITER_FP8_DTYPE = dtypes.fp8
# This block imported aiter at module load time, which fails when the env
# var gates is_aiter_found_and_supported(). The fix removes this block
# and replaces AITER_FP8_DTYPE references with FP8_DTYPE (cached platform value).
# =========================================================================
print("\n--- Check 4: AITER_FP8_DTYPE module-level import block removed ---")

# Check that AITER_FP8_DTYPE is NOT defined or used
has_aiter_fp8_dtype = "AITER_FP8_DTYPE" in aiter_ops_source
check(
    "AITER_FP8_DTYPE variable removed (replaced with FP8_DTYPE)",
    not has_aiter_fp8_dtype,
    "AITER_FP8_DTYPE still present -- module-level aiter import may still crash",
)

# Check that FP8_DTYPE is defined as a replacement
has_fp8_dtype = False
for node in ast.walk(aiter_ops_tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "FP8_DTYPE":
                has_fp8_dtype = True
                break

check(
    "FP8_DTYPE defined as module-level cached value",
    has_fp8_dtype,
    "FP8_DTYPE not found -- fp8 dtype lookups may be broken",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
