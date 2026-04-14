#!/usr/bin/env python3
"""Test harness for vllm ROCm MLA decode backend head count handling.

Tests (behavioral):
  1. Import and instantiate the decode metadata class with a small head count,
     verify the internal head count attribute is safe for the kernel.
  2. Regression guard — large head counts should pass through unchanged.
  3. Anti-hack guard — very small head counts must also produce a safe value.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

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


def run_subprocess(script, timeout=120):
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-mla-nhead-fix test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Behavioral test: instantiate AiterMLADecodeMetadata with controlled
# num_heads values and check the resulting _num_attention_heads attribute.
# ---------------------------------------------------------------------------
print("\n--- Behavioral: head count handling via class instantiation ---")

behavioral_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

# We need to construct a minimal mock that lets us instantiate the class
# and inspect _num_attention_heads without needing a full GPU setup.
# Import the module and patch at the class level.
try:
    import importlib
    import types

    # Import the target module
    spec = importlib.util.spec_from_file_location(
        "rocm_aiter_mla",
        "/workspace/vllm/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
    )
    mod = importlib.util.module_from_spec(spec)

    # Pre-populate sys.modules with stubs for missing dependencies
    # so the module can be loaded without GPU
    stub_modules = [
        "aiter", "aiter.ops", "aiter.ops.mla",
        "aiter.ops.mla.mla_decode_fwd",
        "vllm.v1.attention.backends.mla",
        "vllm.v1.attention.backends.mla.common",
    ]
    for m in stub_modules:
        if m not in sys.modules:
            sys.modules[m] = types.ModuleType(m)

    # Try to load and extract the class's __init__ logic
    # by reading the source and evaluating the assignment expression
    with open("/workspace/vllm/vllm/v1/attention/backends/mla/rocm_aiter_mla.py") as f:
        source = f.read()

    import ast
    tree = ast.parse(source)

    # Find the _num_attention_heads assignment in __init__
    nhead_assign = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if (isinstance(target, ast.Attribute)
                                and target.attr == "_num_attention_heads"
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"):
                            nhead_assign = child
                            break
                if nhead_assign:
                    break
        if nhead_assign:
            break

    if nhead_assign is None:
        print("ASSIGN_NOT_FOUND")
        sys.exit(0)

    rhs_src = ast.get_source_segment(source, nhead_assign.value)
    if rhs_src is None:
        src_lines = source.splitlines()
        assign_lines = src_lines[nhead_assign.lineno - 1:nhead_assign.end_lineno]
        full_assign = "\\n".join(assign_lines)
        eq_idx = full_assign.index("=")
        rhs_src = full_assign[eq_idx + 1:].strip()

    # Evaluate with mock self objects at runtime
    for num_heads in [1, 4, 8, 16, 32, 64]:
        class MockSelf:
            pass
        mock = MockSelf()
        mock.num_heads = num_heads
        try:
            result = eval(compile(rhs_src, "<test>", "eval"),
                          {"self": mock, "max": max, "min": min})
            print(f"NHEAD_{num_heads}:{result}")
        except NameError as e:
            print(f"NHEAD_{num_heads}:NAME_ERROR:{e}")
        except Exception as e:
            print(f"NHEAD_{num_heads}:ERROR:{type(e).__name__}:{e}")

except Exception as e:
    print(f"SETUP_ERROR:{type(e).__name__}:{e}")
"""

stdout, stderr, rc = run_subprocess(behavioral_script, timeout=60)

if "SETUP_ERROR:" in stdout:
    err = stdout.split("SETUP_ERROR:")[1].split("\n")[0]
    check("Module setup", False, err)
elif "ASSIGN_NOT_FOUND" in stdout:
    check("_num_attention_heads assignment found", False, "not found in __init__")
else:
    # Parse results for each test value
    results = {}
    for line in stdout.splitlines():
        if line.startswith("NHEAD_"):
            parts = line.split(":", 1)
            key = parts[0]  # e.g., "NHEAD_8"
            val = parts[1]  # e.g., "16" or "NAME_ERROR:..."
            num_heads = int(key.split("_")[1])
            try:
                results[num_heads] = int(val)
            except ValueError:
                results[num_heads] = val

    # Check: small head counts must produce a value >= 16
    for nh in [1, 4, 8]:
        if nh in results:
            val = results[nh]
            if isinstance(val, int):
                check(
                    f"num_heads={nh} produces safe value (>= 16)",
                    val >= 16,
                    f"got {val} — too small for kernel buffer",
                )
            else:
                check(f"num_heads={nh} evaluates without error", False, str(val))

    # Check: large head counts pass through unchanged
    for nh in [32, 64]:
        if nh in results:
            val = results[nh]
            if isinstance(val, int):
                check(
                    f"num_heads={nh} passes through unchanged",
                    val == nh,
                    f"got {val} — large value should not be modified",
                )
            else:
                check(f"num_heads={nh} evaluates without error", False, str(val))

    # Check: num_heads=16 boundary — should be exactly 16
    if 16 in results:
        val = results[16]
        if isinstance(val, int):
            check(
                f"num_heads=16 boundary handled correctly",
                val == 16,
                f"got {val}",
            )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
