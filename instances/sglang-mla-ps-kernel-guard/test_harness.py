#!/usr/bin/env python3
"""Test harness for sglang-mla-ps-kernel-guard.

Verify that MLA-specific code paths are properly guarded so non-MLA
models can run without errors.

The bug: non-MLA models crash with
  AttributeError: 'AiterAttnBackend' object has no attribute 'max_split_per_batch'
because MLA-specific attributes are accessed unconditionally in the
CUDA graph init methods.
"""
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
print("sglang-mla-ps-kernel-guard test harness")
print("=" * 60)

_PY = "/opt/venv/bin/python3"


def run_subprocess(script, timeout=120):
    result = subprocess.run(
        [_PY, "-c", script],
        capture_output=True, text=True, timeout=timeout, cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


# ---------------------------------------------------------------------------
# Check 1: Module can be imported
# ---------------------------------------------------------------------------
print("\n--- Check 1: Import ---")
import_script = textwrap.dedent("""\
    import sys
    sys.path.insert(0, '/workspace/sglang/python')
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    print('IMPORT_OK')
""")
stdout, stderr, rc = run_subprocess(import_script)
import_ok = "IMPORT_OK" in stdout

if not import_ok:
    check("AiterAttnBackend importable", False,
          f"Import failed: {stderr[-300:]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("AiterAttnBackend importable", True)


# ---------------------------------------------------------------------------
# Check 2: AiterAttnBackend has CUDA graph init methods
# ---------------------------------------------------------------------------
print("\n--- Check 2: Class structure ---")
class_script = textwrap.dedent("""\
    import sys
    sys.path.insert(0, '/workspace/sglang/python')
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    methods = dir(AiterAttnBackend)
    has_capture = 'init_forward_metadata_capture_cuda_graph' in methods
    has_replay = 'init_forward_metadata_replay_cuda_graph' in methods
    print(f'CAPTURE={has_capture},REPLAY={has_replay}')
""")
stdout2, stderr2, rc2 = run_subprocess(class_script)
has_capture = "CAPTURE=True" in stdout2
has_replay = "REPLAY=True" in stdout2

check("CUDA graph capture method exists", has_capture,
      "init_forward_metadata_capture_cuda_graph not found")
check("CUDA graph replay method exists", has_replay,
      "init_forward_metadata_replay_cuda_graph not found")


# ---------------------------------------------------------------------------
# Check 3 (PRIMARY): Non-MLA model does not crash with AttributeError
# on max_split_per_batch when calling CUDA graph init methods.
#
# Uses a permissive mock that simulates a non-MLA backend instance:
# - use_mla = False
# - max_split_per_batch deliberately not set (only MLA models set it)
# - All other attributes return permissive mock values
#
# Pre-fix: method enters _use_mla_ps_kernel block without checking
#   use_mla, accesses self.max_split_per_batch -> AttributeError
# Post-fix: method checks use_mla before entering MLA block, skips it
#   -> no AttributeError (may crash on other mocked values, which is OK)
# ---------------------------------------------------------------------------
print("\n--- Check 3: Non-MLA compatibility ---")
non_mla_script = textwrap.dedent("""\
    import sys, inspect
    sys.path.insert(0, '/workspace/sglang/python')

    from unittest.mock import MagicMock, PropertyMock

    try:
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        import sglang.srt.layers.attention.aiter_backend as mod

        # Ensure _use_mla_ps_kernel is True (the default that triggers the bug)
        if hasattr(mod, '_use_mla_ps_kernel'):
            mod._use_mla_ps_kernel = True

        # Create mock with use_mla=False
        # max_split_per_batch raises AttributeError (simulates non-MLA model)
        mock = MagicMock()
        mock.use_mla = False
        mock.sliding_window_size = -1
        type(mock).max_split_per_batch = PropertyMock(
            side_effect=AttributeError(
                "'AiterAttnBackend' object has no attribute 'max_split_per_batch'"
            )
        )

        bugs = []

        for method_name in [
            'init_forward_metadata_capture_cuda_graph',
            'init_forward_metadata_replay_cuda_graph',
        ]:
            method = getattr(AiterAttnBackend, method_name, None)
            if method is None:
                continue

            # Determine correct argument count from signature
            sig = inspect.signature(method)
            params = [p for p in sig.parameters.keys() if p != 'self']
            args = [MagicMock() for _ in params]

            try:
                method(mock, *args)
            except AttributeError as e:
                if 'max_split_per_batch' in str(e):
                    bugs.append(method_name.replace(
                        'init_forward_metadata_', '').replace('_cuda_graph', ''))
            except Exception:
                pass  # Non-AttributeError means the guard worked

        if bugs:
            print('BUG_PRESENT:' + ','.join(bugs))
        else:
            print('BUG_ABSENT')

    except ImportError as e:
        print(f'IMPORT_FAIL:{e}')
    except Exception as e:
        print(f'ERROR:{type(e).__name__}:{e}')
""")
stdout3, stderr3, rc3 = run_subprocess(non_mla_script)

if "BUG_ABSENT" in stdout3:
    check("Non-MLA model: no max_split_per_batch crash", True)
elif "BUG_PRESENT" in stdout3:
    detail = stdout3.split("BUG_PRESENT:")[-1].strip()
    check("Non-MLA model: no max_split_per_batch crash", False,
          f"MLA-specific attribute accessed without guard in: {detail}")
elif "IMPORT_FAIL" in stdout3:
    check("Non-MLA model: no max_split_per_batch crash", False,
          f"Module import failed: {stdout3}")
else:
    check("Non-MLA model: no max_split_per_batch crash", False,
          f"Unexpected: stdout={stdout3[:200]}, stderr={stderr3[:200]}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
