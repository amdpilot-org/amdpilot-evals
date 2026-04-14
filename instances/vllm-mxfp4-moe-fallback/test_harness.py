#!/usr/bin/env python3
"""Test harness for vllm-mxfp4-moe-fallback (PR #35893). Behavioral tests only.

Bug: CK MXFP4 MoE GEMM kernels crash with RuntimeError when intermediate_size
     per partition is not a multiple of 256 (e.g. MiniMax-M2.1 TP=4 → 384).
     No dimension validation or fallback exists.

Expected behavior after fix: When CK alignment fails, the code gracefully
     falls back to Triton backend (Mxfp4MoEMethod) or emulation mode
     (QuarkOCP_MX_MoEMethod) instead of crashing.
"""
import sys
import subprocess

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


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace/vllm",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-mxfp4-moe-fallback test harness")
print("=" * 60)

# -----------------------------------------------------------------------
# Check 1: Mxfp4MoEMethod construction with misaligned dimensions does
#           not crash and falls back to a non-CK backend.
#
# Strategy: Create a FusedMoEConfig with intermediate_size_per_partition=384
# (not a multiple of 256), force CK backend selection, and construct
# Mxfp4MoEMethod. Pre-fix: crashes with RuntimeError or proceeds with
# CK (which would crash at GEMM time). Post-fix: falls back to Triton.
# -----------------------------------------------------------------------
print("\n--- Check 1: Mxfp4MoEMethod CK → Triton fallback ---")

stdout, stderr, rc = run_test("""
import sys
sys.path.insert(0, '/workspace/vllm')

try:
    from unittest.mock import patch, MagicMock
    import importlib

    # Import the mxfp4 module
    try:
        mxfp4_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.mxfp4")
    except ImportError as e:
        print(f"IMPORT_SKIP:mxfp4:{e}")
        sys.exit(0)
    except Exception as e:
        print(f"IMPORT_SKIP:mxfp4:{type(e).__name__}:{e}")
        sys.exit(0)

    # Check if Mxfp4MoEMethod exists
    Mxfp4MoEMethod = getattr(mxfp4_mod, "Mxfp4MoEMethod", None)
    if Mxfp4MoEMethod is None:
        print("IMPORT_SKIP:no_Mxfp4MoEMethod")
        sys.exit(0)

    # Check if Mxfp4Backend enum exists
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            Mxfp4Backend)
    except ImportError:
        Mxfp4Backend = getattr(mxfp4_mod, "Mxfp4Backend", None)
    if Mxfp4Backend is None:
        print("IMPORT_SKIP:no_Mxfp4Backend")
        sys.exit(0)

    # Create a mock FusedMoEConfig with misaligned dimension
    # 384 is NOT a multiple of 256 — this is the trigger
    moe_config = MagicMock()
    moe_config.intermediate_size_per_partition = 384
    moe_config.num_experts = 8
    moe_config.top_k = 2
    moe_config.intermediate_size = 1536
    moe_config.hidden_size = 512

    # Patch backend selection to force CK, and triton availability to True
    get_backend_path = None
    for attr in dir(mxfp4_mod):
        obj = getattr(mxfp4_mod, attr, None)
        if callable(obj) and 'backend' in attr.lower():
            get_backend_path = f"vllm.model_executor.layers.quantization.mxfp4.{attr}"
            break

    # Try multiple approaches to force the CK backend path
    try:
        # Approach 1: Direct construction with mocks
        with patch.object(mxfp4_mod, "get_mxfp4_backend",
                          return_value=getattr(Mxfp4Backend, "CK", 7),
                          create=True):
            try:
                # Also ensure has_triton_kernels returns True for fallback
                with patch.object(mxfp4_mod, "has_triton_kernels",
                                  return_value=True, create=True):
                    method = Mxfp4MoEMethod(moe_config)
            except TypeError:
                # May need different constructor args
                method = Mxfp4MoEMethod(moe_config)

        # Check which backend was selected
        backend = getattr(method, "mxfp4_backend", None)
        if backend is None:
            backend = getattr(method, "backend", None)

        # The fix should have changed backend from CK to TRITON
        triton_val = getattr(Mxfp4Backend, "TRITON", 6)
        ck_val = getattr(Mxfp4Backend, "CK", 7)

        if backend == triton_val or (hasattr(backend, 'name') and
                                      backend.name == "TRITON"):
            print("MXFP4_FALLBACK:TRITON")
        elif backend == ck_val or (hasattr(backend, 'name') and
                                    backend.name == "CK"):
            print("MXFP4_FALLBACK:STILL_CK")
        else:
            print(f"MXFP4_FALLBACK:UNKNOWN:{backend}")

        print("MXFP4_CONSTRUCT:OK")

    except (RuntimeError, ValueError) as e:
        print(f"MXFP4_CONSTRUCT:CRASH:{type(e).__name__}:{str(e)[:200]}")
    except Exception as e:
        print(f"MXFP4_CONSTRUCT:ERROR:{type(e).__name__}:{str(e)[:200]}")

except Exception as e:
    print(f"OUTER_ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout:
    err = stdout.split("IMPORT_SKIP:")[1].split("\n")[0].strip()
    check("Mxfp4MoEMethod handles misaligned dimensions without crash",
          False, f"import failed: {err}")
    check("Mxfp4MoEMethod falls back from CK to Triton backend",
          False, "import failed")
elif "MXFP4_CONSTRUCT:CRASH" in stdout:
    err = stdout.split("MXFP4_CONSTRUCT:CRASH:")[1].split("\n")[0].strip()
    check("Mxfp4MoEMethod handles misaligned dimensions without crash",
          False, f"construction crashed: {err}")
    check("Mxfp4MoEMethod falls back from CK to Triton backend",
          False, "construction crashed — no fallback logic")
elif "MXFP4_CONSTRUCT:OK" in stdout:
    check("Mxfp4MoEMethod handles misaligned dimensions without crash", True)
    fallback_ok = "MXFP4_FALLBACK:TRITON" in stdout
    check("Mxfp4MoEMethod falls back from CK to Triton backend",
          fallback_ok,
          f"backend not changed to TRITON: {stdout.strip()[:200]}")
else:
    err = stdout.strip()[:200] if stdout.strip() else stderr.strip()[:200]
    check("Mxfp4MoEMethod handles misaligned dimensions without crash",
          False, f"unexpected: {err}")
    check("Mxfp4MoEMethod falls back from CK to Triton backend",
          False, f"unexpected: {err}")


# -----------------------------------------------------------------------
# Check 2: Mxfp4MoEMethod with ALIGNED dimensions still uses CK.
#           (No regression — aligned dims should NOT trigger fallback.)
# -----------------------------------------------------------------------
print("\n--- Check 2: Aligned dimensions preserve CK backend ---")

stdout2, stderr2, rc2 = run_test("""
import sys
sys.path.insert(0, '/workspace/vllm')

try:
    from unittest.mock import patch, MagicMock
    import importlib

    mxfp4_mod = importlib.import_module(
        "vllm.model_executor.layers.quantization.mxfp4")
    Mxfp4MoEMethod = getattr(mxfp4_mod, "Mxfp4MoEMethod", None)
    if Mxfp4MoEMethod is None:
        print("IMPORT_SKIP:no_Mxfp4MoEMethod")
        sys.exit(0)

    try:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            Mxfp4Backend)
    except ImportError:
        Mxfp4Backend = getattr(mxfp4_mod, "Mxfp4Backend", None)
    if Mxfp4Backend is None:
        print("IMPORT_SKIP:no_Mxfp4Backend")
        sys.exit(0)

    # Create a mock FusedMoEConfig with ALIGNED dimension (512 = 2*256)
    moe_config = MagicMock()
    moe_config.intermediate_size_per_partition = 512
    moe_config.num_experts = 8
    moe_config.top_k = 2
    moe_config.intermediate_size = 2048
    moe_config.hidden_size = 512

    ck_val = getattr(Mxfp4Backend, "CK", 7)
    triton_val = getattr(Mxfp4Backend, "TRITON", 6)

    with patch.object(mxfp4_mod, "get_mxfp4_backend",
                      return_value=ck_val, create=True):
        try:
            with patch.object(mxfp4_mod, "has_triton_kernels",
                              return_value=True, create=True):
                method = Mxfp4MoEMethod(moe_config)
        except TypeError:
            method = Mxfp4MoEMethod(moe_config)

    backend = getattr(method, "mxfp4_backend", None)
    if backend is None:
        backend = getattr(method, "backend", None)

    if backend == ck_val or (hasattr(backend, 'name') and
                              backend.name == "CK"):
        print("ALIGNED_BACKEND:CK")
    elif backend == triton_val or (hasattr(backend, 'name') and
                                    backend.name == "TRITON"):
        print("ALIGNED_BACKEND:TRITON")
    else:
        print(f"ALIGNED_BACKEND:UNKNOWN:{backend}")

    print("ALIGNED_CONSTRUCT:OK")

except Exception as e:
    print(f"ALIGNED_ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout2:
    check("Aligned dimensions preserve CK backend (no regression)",
          False, "import failed")
elif "ALIGNED_CONSTRUCT:OK" in stdout2:
    ck_preserved = "ALIGNED_BACKEND:CK" in stdout2
    check("Aligned dimensions preserve CK backend (no regression)",
          ck_preserved,
          f"expected CK for aligned dims: {stdout2.strip()[:200]}")
else:
    check("Aligned dimensions preserve CK backend (no regression)",
          False, f"error: {(stdout2 + stderr2).strip()[:200]}")


# -----------------------------------------------------------------------
# Check 3: QuarkOCP_MX_MoEMethod with misaligned dimensions falls back
#           to emulation mode (self.emulate = True).
# -----------------------------------------------------------------------
print("\n--- Check 3: QuarkOCP_MX_MoEMethod emulation fallback ---")

stdout3, stderr3, rc3 = run_test("""
import sys
sys.path.insert(0, '/workspace/vllm')

try:
    from unittest.mock import patch, MagicMock
    import importlib

    try:
        quark_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.quark.quark_moe")
    except ImportError as e:
        print(f"IMPORT_SKIP:quark_moe:{e}")
        sys.exit(0)
    except Exception as e:
        print(f"IMPORT_SKIP:quark_moe:{type(e).__name__}:{e}")
        sys.exit(0)

    QuarkOCP_MX_MoEMethod = getattr(quark_mod, "QuarkOCP_MX_MoEMethod", None)
    if QuarkOCP_MX_MoEMethod is None:
        # Try alternative names
        for name in dir(quark_mod):
            obj = getattr(quark_mod, name, None)
            if isinstance(obj, type) and 'quark' in name.lower() and 'moe' in name.lower():
                if 'mx' in name.lower() or 'ocp' in name.lower():
                    QuarkOCP_MX_MoEMethod = obj
                    break
    if QuarkOCP_MX_MoEMethod is None:
        print("IMPORT_SKIP:no_QuarkOCP_MX_MoEMethod")
        sys.exit(0)

    # Create mock configs
    weight_config = MagicMock()
    weight_config.num_bits = 4
    input_config = MagicMock()
    input_config.num_bits = 4

    moe_config = MagicMock()
    moe_config.intermediate_size_per_partition = 384  # misaligned
    moe_config.num_experts = 8
    moe_config.top_k = 2
    moe_config.intermediate_size = 1536
    moe_config.hidden_size = 512

    import inspect
    sig = inspect.signature(QuarkOCP_MX_MoEMethod.__init__)
    params = list(sig.parameters.keys())

    # Build constructor args dynamically
    kwargs = {}
    for p in params:
        if p == 'self': continue
        pl = p.lower()
        if 'weight' in pl and 'config' in pl: kwargs[p] = weight_config
        elif 'input' in pl and 'config' in pl: kwargs[p] = input_config
        elif 'moe' in pl or 'fused' in pl: kwargs[p] = moe_config
        elif 'quant' in pl and 'config' in pl: kwargs[p] = MagicMock()

    try:
        # Set up environment so CK path is initially selected
        with patch.object(quark_mod, "is_hip", return_value=True, create=True):
            method = QuarkOCP_MX_MoEMethod(**kwargs)
    except TypeError:
        # Try with positional args
        try:
            method = QuarkOCP_MX_MoEMethod(weight_config, input_config, moe_config)
        except Exception as e:
            print(f"QUARK_CONSTRUCT:ERROR:{type(e).__name__}:{str(e)[:200]}")
            sys.exit(0)
    except Exception as e:
        print(f"QUARK_CONSTRUCT:ERROR:{type(e).__name__}:{str(e)[:200]}")
        sys.exit(0)

    emulate = getattr(method, "emulate", None)
    use_aiter = getattr(method, "use_rocm_aiter_moe", None)

    print(f"QUARK_EMULATE:{emulate}")
    print(f"QUARK_USE_AITER:{use_aiter}")
    print("QUARK_CONSTRUCT:OK")

except Exception as e:
    print(f"QUARK_OUTER_ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_SKIP" in stdout3:
    err = stdout3.split("IMPORT_SKIP:")[1].split("\n")[0].strip()
    check("QuarkOCP_MX_MoEMethod handles misaligned dims",
          False, f"import failed: {err}")
    check("QuarkOCP_MX_MoEMethod falls back to emulation mode",
          False, "import failed")
elif "QUARK_CONSTRUCT:ERROR" in stdout3:
    err = stdout3.split("QUARK_CONSTRUCT:ERROR:")[1].split("\n")[0].strip()
    check("QuarkOCP_MX_MoEMethod handles misaligned dims",
          False, f"construction failed: {err}")
    check("QuarkOCP_MX_MoEMethod falls back to emulation mode",
          False, "construction failed")
elif "QUARK_CONSTRUCT:OK" in stdout3:
    check("QuarkOCP_MX_MoEMethod handles misaligned dims", True)

    # Post-fix: emulate should be True, use_rocm_aiter_moe should be False
    emulate_ok = "QUARK_EMULATE:True" in stdout3
    aiter_off = "QUARK_USE_AITER:False" in stdout3

    check("QuarkOCP_MX_MoEMethod falls back to emulation mode",
          emulate_ok or aiter_off,
          f"emulate={emulate_ok}, use_aiter_off={aiter_off} — "
          f"expected emulate=True or use_rocm_aiter_moe=False")
else:
    err = (stdout3 + stderr3).strip()[:200]
    check("QuarkOCP_MX_MoEMethod handles misaligned dims",
          False, f"unexpected: {err}")
    check("QuarkOCP_MX_MoEMethod falls back to emulation mode",
          False, f"unexpected: {err}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
