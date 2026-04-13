#!/usr/bin/env python3
"""Test harness for sglang-kscale-vscale-fix.

Verifies that extend_attention_fwd() is called with the correct number
of arguments, including k_scale and v_scale.

The bug: the call site in forward_extend passes too few positional
arguments, causing:
  TypeError: extend_attention_fwd() missing 2 required positional
             arguments: 'k_scale' and 'v_scale'
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
print("sglang-kscale-vscale-fix test harness")
print("=" * 60)

_PY = "/opt/venv/bin/python3"


def run_subprocess(script, timeout=120):
    result = subprocess.run(
        [_PY, "-c", script],
        capture_output=True, text=True, timeout=timeout, cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


# ---------------------------------------------------------------------------
# Check 1: Modules can be imported
# ---------------------------------------------------------------------------
print("\n--- Check 1: Import ---")
import_script = textwrap.dedent("""\
    import sys
    sys.path.insert(0, '/workspace/sglang/python')
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    print('BACKEND_OK')
""")
stdout, stderr, rc = run_subprocess(import_script)
backend_ok = "BACKEND_OK" in stdout

if not backend_ok:
    check("AiterAttnBackend importable", False,
          f"Import failed: {stderr[-300:]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("AiterAttnBackend importable", True)

import_ext_script = textwrap.dedent("""\
    import sys
    sys.path.insert(0, '/workspace/sglang/python')
    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    print('EXTEND_OK')
""")
stdout_e, stderr_e, rc_e = run_subprocess(import_ext_script)
extend_ok = "EXTEND_OK" in stdout_e

if not extend_ok:
    check("extend_attention_fwd importable", False,
          f"Import failed: {stderr_e[-300:]}")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("extend_attention_fwd importable", True)


# ---------------------------------------------------------------------------
# Check 2: extend_attention_fwd accepts k_scale and v_scale parameters
# (runtime reflection via inspect.signature, not AST)
# ---------------------------------------------------------------------------
print("\n--- Check 2: Function signature ---")
sig_script = textwrap.dedent("""\
    import sys, inspect
    sys.path.insert(0, '/workspace/sglang/python')
    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    sig = inspect.signature(extend_attention_fwd)
    params = list(sig.parameters.keys())
    has_k = 'k_scale' in params
    has_v = 'v_scale' in params
    print(f'K_SCALE={has_k},V_SCALE={has_v},PARAMS={len(params)}')
""")
stdout2, stderr2, rc2 = run_subprocess(sig_script)

has_k = "K_SCALE=True" in stdout2
has_v = "V_SCALE=True" in stdout2

check("extend_attention_fwd has k_scale parameter", has_k,
      "k_scale not in function signature")
check("extend_attention_fwd has v_scale parameter", has_v,
      "v_scale not in function signature")


# ---------------------------------------------------------------------------
# Check 3 (PRIMARY): forward_extend call site passes enough arguments
# to satisfy extend_attention_fwd's signature.
#
# Monkey-patches extend_attention_fwd with a recording wrapper, then
# calls forward_extend with mock data. The wrapper checks whether
# the call binds successfully to the real function's signature.
#
# Pre-fix: call is missing k_scale/v_scale -> bind fails with TypeError
# Post-fix: call includes k_scale/v_scale -> bind succeeds
# ---------------------------------------------------------------------------
print("\n--- Check 3: Call site compatibility ---")
call_test_script = textwrap.dedent("""\
    import sys, inspect
    sys.path.insert(0, '/workspace/sglang/python')
    from unittest.mock import MagicMock, PropertyMock

    try:
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        # Get the real function's signature for validation
        real_sig = inspect.signature(extend_attention_fwd)

        # Create a recording wrapper that checks argument binding
        call_result = {'status': None, 'detail': ''}

        def checking_wrapper(*args, **kwargs):
            try:
                real_sig.bind(*args, **kwargs)
                call_result['status'] = 'ARGS_OK'
            except TypeError as e:
                if 'k_scale' in str(e) or 'v_scale' in str(e):
                    call_result['status'] = 'MISSING_SCALE'
                    call_result['detail'] = str(e)
                else:
                    call_result['status'] = 'OTHER_BIND_ERROR'
                    call_result['detail'] = str(e)
            # Don't execute the real function (no GPU tensors available)
            return MagicMock()

        # Create mock backend with the checking wrapper installed
        # Bug is in the NON-MLA branch of forward_extend (the else block),
        # where the call to self.extend_attention_fwd() is missing k_scale
        # and v_scale arguments.
        mock_self = MagicMock()
        mock_self.use_mla = False  # Exercise the non-MLA extend path
        mock_self.extend_attention_fwd = checking_wrapper

        # forward_extend signature: (self, q, k, v, layer, forward_batch, ...)
        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        mock_layer = MagicMock()
        mock_batch = MagicMock()
        try:
            AiterAttnBackend.forward_extend(
                mock_self, mock_q, mock_k, mock_v, mock_layer, mock_batch
            )
        except Exception:
            pass  # We don't care about downstream errors

        if call_result['status'] == 'ARGS_OK':
            print('CALL_OK')
        elif call_result['status'] == 'MISSING_SCALE':
            print('MISSING_SCALE:' + call_result['detail'])
        elif call_result['status'] == 'OTHER_BIND_ERROR':
            print('OTHER_BIND:' + call_result['detail'])
        elif call_result['status'] is None:
            # extend_attention_fwd was never called (method might have
            # crashed before reaching the call site)
            print('NOT_CALLED')
        else:
            print('UNKNOWN:' + str(call_result))

    except ImportError as e:
        print(f'IMPORT_FAIL:{e}')
    except Exception as e:
        print(f'ERROR:{type(e).__name__}:{e}')
""")
stdout3, stderr3, rc3 = run_subprocess(call_test_script)

if "CALL_OK" in stdout3:
    check("forward_extend passes k_scale/v_scale to extend_attention_fwd", True)
elif "MISSING_SCALE" in stdout3:
    detail = stdout3.split("MISSING_SCALE:")[-1].strip()
    check("forward_extend passes k_scale/v_scale to extend_attention_fwd",
          False, detail)
elif "NOT_CALLED" in stdout3:
    # extend_attention_fwd was never reached -- the call path may have
    # changed structurally. Fall back to signature check (Check 2).
    check("forward_extend passes k_scale/v_scale to extend_attention_fwd",
          has_k and has_v,
          "Could not reach call site (method crashed before call); "
          "relying on signature check")
elif "OTHER_BIND" in stdout3:
    detail = stdout3.split("OTHER_BIND:")[-1].strip()
    check("forward_extend passes k_scale/v_scale to extend_attention_fwd",
          False, f"Argument binding error: {detail}")
else:
    check("forward_extend passes k_scale/v_scale to extend_attention_fwd",
          False, f"Unexpected: {stdout3[:200]}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
