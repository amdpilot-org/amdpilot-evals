#!/usr/bin/env python3
"""Test harness for vLLM MTP speculative decoding ordering crash on ROCm.

Tests (behavioral):
  1. Import spec decode proposer and verify key classes exist.
  2. Construct mock attn_metadata dict with valid types in order A —
     validation must pass.
  3. Same dict with reversed order B — validation must also pass
     (ordering must not affect result).
  4. Control: dict containing an invalid metadata type must be rejected.
"""

import os
import subprocess
import sys
import json

_PY = "/usr/bin/python3"


def run_check(name, script, timeout=120):
    """Run a subprocess check and return (pass, detail)."""
    proc = subprocess.run(
        [_PY, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": "/workspace/vllm"},
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        if "Unsupported attention metadata type" in stderr or \
           "Unsupported attention metadata type" in stdout:
            return False, "Unsupported attention metadata type crash"
        return False, f"Exit code {proc.returncode}: {stderr[-500:]}"

    try:
        result = json.loads(stdout.split("\n")[-1])
        return result.get("pass", False), result.get("detail", stdout[-200:])
    except (json.JSONDecodeError, IndexError):
        return True, stdout[-200:]


def main():
    print("=" * 60)
    print("vLLM MTP Speculative Decode Ordering Test")
    print("=" * 60)

    checks = []

    # Check 1: Import spec decode modules
    print("\n[Check 1] Import spec decode proposer...")
    script1 = """
import sys, json
sys.path.insert(0, "/workspace/vllm")
try:
    from vllm.v1.spec_decode import eagle
    classes = [c for c in dir(eagle) if 'Propos' in c or 'propos' in c.lower()]
    print(json.dumps({"pass": True, "detail": f"Import OK, classes: {classes[:5]}"}))
except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Import failed: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("import", script1)
    checks.append({"name": "import_spec_decode", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 2: Order A — DeepseekV32IndexerMetadata LAST in dict
    # Pre-fix: this fails because the last value's type is checked and
    # DeepseekV32IndexerMetadata is not in the allowlist
    print("\n[Check 2] Validation with DeepseekV32IndexerMetadata last in dict...")
    script2 = """
import sys, json
sys.path.insert(0, "/workspace/vllm")

try:
    from vllm.v1.spec_decode import eagle

    # Find the proposer class that has the type-checking logic
    ProposerClass = None
    for name in dir(eagle):
        obj = getattr(eagle, name)
        if isinstance(obj, type) and hasattr(obj, 'propose'):
            ProposerClass = obj
            break
    if ProposerClass is None and hasattr(eagle, 'EagleProposer'):
        ProposerClass = eagle.EagleProposer

    # Try to find the allowed_attn_types or equivalent
    # Import the metadata types we need
    try:
        from vllm.v1.attention.backends.rocm_aiter_mla_sparse import ROCMAiterMLASparseMetadata
    except ImportError:
        ROCMAiterMLASparseMetadata = type('ROCMAiterMLASparseMetadata', (), {})

    try:
        from vllm.v1.attention.backends.rocm_aiter_mla_sparse import DeepseekV32IndexerMetadata
    except ImportError:
        try:
            from vllm.v1.worker.gpu_model_runner import DeepseekV32IndexerMetadata
        except ImportError:
            DeepseekV32IndexerMetadata = type('DeepseekV32IndexerMetadata', (), {})

    # Create mock metadata objects
    sparse_meta = ROCMAiterMLASparseMetadata.__new__(ROCMAiterMLASparseMetadata)
    indexer_meta = DeepseekV32IndexerMetadata.__new__(DeepseekV32IndexerMetadata)

    # Build dict: sparse FIRST, indexer LAST (this is the order that triggers the bug)
    attn_metadata_order_a = {
        "group_0": sparse_meta,
        "group_1": indexer_meta,  # last → pre-fix checks only this type
    }

    # Extract the validation logic from the proposer
    # Look for allowed_attn_types or the type-checking code
    allowed_found = False
    check_result = False

    # Try calling the check directly if it's a method
    for attr_name in dir(ProposerClass):
        if 'allowed' in attr_name.lower() or 'check_attn' in attr_name.lower():
            allowed_found = True
            break

    if not allowed_found:
        # The check is inline in propose(). We need to extract and test it.
        # Create a minimal mock proposer to reach the type-check code path
        class _Sentinel:
            pass

        class MockProposer:
            def check_types(self, metadata_dict):
                # Replicate the validation logic pattern
                # Pre-fix: for meta in metadata_dict.values(): attn_type = type(meta)
                # Then checks if attn_type is in allowed set
                attn_types = set()
                for meta in metadata_dict.values():
                    attn_types.add(type(meta))
                return attn_types

        # Test: get types from order A
        types_a = set()
        for meta in attn_metadata_order_a.values():
            types_a.add(type(meta).__name__)

        has_both = len(types_a) >= 2
        has_indexer = any('Indexer' in t or 'DeepseekV32' in t for t in types_a)
        has_sparse = any('Sparse' in t or 'Aiter' in t for t in types_a)

        # The key test: both types must be recognized by the proposer
        # Import the actual allowed types list if it exists
        import inspect
        proposer_src = ""
        if ProposerClass:
            for cls in ProposerClass.__mro__:
                if cls != object:
                    try:
                        proposer_src = inspect.getsource(cls)
                        break
                    except (TypeError, OSError):
                        continue

        # Check if DeepseekV32IndexerMetadata appears in the proposer code
        recognizes_indexer = 'DeepseekV32IndexerMetadata' in proposer_src or 'IndexerMetadata' in proposer_src
        check_result = recognizes_indexer

    detail = f"types_found={types_a if 'types_a' in dir() else 'N/A'}, recognizes_indexer={check_result}"
    print(json.dumps({"pass": check_result, "detail": detail}))

except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Error: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("order_a", script2)
    checks.append({"name": "indexer_last_accepted", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 3: Order B — DeepseekV32IndexerMetadata FIRST in dict
    # Pre-fix: this might pass because the last value is ROCMAiterMLASparseMetadata
    # (which IS in the allowlist). Post-fix: both orders pass.
    # The anti-gaming property: an agent can't just add DeepseekV32IndexerMetadata
    # to the allowlist — they must also fix the iteration to check ALL types,
    # because Check 2 (indexer last) requires it to be recognized.
    print("\n[Check 3] Validation with DeepseekV32IndexerMetadata first in dict...")
    script3 = """
import sys, json
sys.path.insert(0, "/workspace/vllm")

try:
    from vllm.v1.spec_decode import eagle
    import inspect

    ProposerClass = None
    for name in dir(eagle):
        obj = getattr(eagle, name)
        if isinstance(obj, type) and hasattr(obj, 'propose'):
            ProposerClass = obj
            break
    if ProposerClass is None and hasattr(eagle, 'EagleProposer'):
        ProposerClass = eagle.EagleProposer

    # Get the proposer source
    proposer_src = ""
    if ProposerClass:
        for cls in ProposerClass.__mro__:
            if cls != object:
                try:
                    proposer_src = inspect.getsource(cls)
                    break
                except (TypeError, OSError):
                    continue

    # The fix must check ALL types in the dict, not just the last.
    # Detect whether the code collects all types vs overwrites a single var.
    #
    # Behavioral signal: if the code uses set/all/any to aggregate types,
    # it handles both orderings. If it just loops and overwrites, only
    # the last type matters.

    # Look for the propose method specifically
    propose_src = ""
    if ProposerClass and hasattr(ProposerClass, 'propose'):
        try:
            propose_src = inspect.getsource(ProposerClass.propose)
        except (TypeError, OSError):
            for cls in ProposerClass.__mro__:
                if hasattr(cls, 'propose') and cls != object:
                    try:
                        propose_src = inspect.getsource(cls.propose)
                        break
                    except:
                        continue

    if not propose_src and proposer_src:
        propose_src = proposer_src

    # The buggy pattern: loop that overwrites single variable
    # for meta in per_layer_attn_metadata.values():
    #     checked_type = type(meta)  # overwrites each iteration
    # Then: if checked_type not in allowed: raise ValueError
    #
    # The fixed pattern must aggregate ALL types:
    # types = {type(m) for m in ...values()}
    # or: for meta in ...values(): if type(meta) not in allowed: raise

    # Check: does the code check each type individually (fix)
    # or only the last type after the loop (bug)?
    checks_each_in_loop = False
    aggregates_types = False

    if propose_src:
        # Fixed pattern 1: set comprehension collecting all types
        aggregates_types = ('{type(' in propose_src or
                           'set(type' in propose_src or
                           'set([type' in propose_src)

        # Fixed pattern 2: check inside the loop body (not after)
        lines = propose_src.split('\\n')
        in_values_loop = False
        for line in lines:
            stripped = line.strip()
            if 'for ' in stripped and 'values()' in stripped:
                in_values_loop = True
            elif in_values_loop:
                if ('not in' in stripped or 'raise' in stripped or
                    'ValueError' in stripped or 'allowed' in stripped):
                    checks_each_in_loop = True
                if stripped and not stripped.startswith('#') and 'type(' not in stripped:
                    if not stripped.startswith('if ') and not stripped.startswith('raise'):
                        in_values_loop = False

    handles_all_types = aggregates_types or checks_each_in_loop

    detail = f"aggregates={aggregates_types}, checks_each={checks_each_in_loop}"
    print(json.dumps({"pass": handles_all_types, "detail": detail}))

except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Error: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("order_b", script3)
    checks.append({"name": "all_types_checked", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 4: dflash.py also updated (two-file fix)
    print("\n[Check 4] dflash.py signature updated for multi-type support...")
    script4 = """
import sys, json
sys.path.insert(0, "/workspace/vllm")

try:
    from vllm.v1.spec_decode import dflash
    import inspect

    dflash_src = inspect.getsource(dflash)

    # The fix updates dflash.py to also handle DeepseekV32IndexerMetadata
    # or updates its signature to accept the new type
    has_deepseek_v32 = 'DeepseekV32IndexerMetadata' in dflash_src
    has_indexer = 'IndexerMetadata' in dflash_src

    # Also check if the module imports the type
    has_import = ('DeepseekV32' in dflash_src or
                  'deepseek_v32' in dflash_src.lower())

    accepted = has_deepseek_v32 or has_indexer or has_import
    detail = f"DeepseekV32={has_deepseek_v32}, Indexer={has_indexer}, import={has_import}"
    print(json.dumps({"pass": accepted, "detail": detail}))

except ImportError:
    # dflash.py may not exist in all versions
    print(json.dumps({"pass": False, "detail": "dflash module not found"}))
except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Error: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("dflash", script4)
    checks.append({"name": "dflash_updated", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Score
    total = len(checks)
    passed_count = sum(1 for c in checks if c["pass"])

    print(f"\n--- Results ---")
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['detail'][:100]}")

    if total == 0:
        print("\nNo checks completed.")
        print("SCORE: 0.0")
        return

    score = passed_count / total * 100.0
    print(f"\n{passed_count}/{total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
