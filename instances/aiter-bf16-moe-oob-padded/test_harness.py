#!/usr/bin/env python3
"""Test harness for aiter-bf16-moe-oob-padded eval instance.

Binary scoring:
  - 100.0 if bench_bf16_moe.sh runs to completion, produces valid decode output,
    and the MoE dispatch path handles out-of-bounds token IDs correctly
  -   0.0 if the benchmark crashes, fails, or produces incorrect output
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time


# Dispatch log directory written by the injected sitecustomize hook.
# Each worker process writes its own file: /tmp/ck_dispatch_logs/<pid>.json
DISPATCH_LOG_DIR = "/tmp/ck_dispatch_logs"

# Directory where the sitecustomize hook is staged.
HOOK_DIR = "/tmp/_ck_dispatch_hook"

# The sitecustomize.py content injected into the benchmark subprocess.
# It monkey-patches CK MoE dispatch functions to record invocation counts,
# then writes the log at exit.
SITECUSTOMIZE_SRC = r'''
import atexit
import builtins
import functools
import json
import os
import sys

_ck_dispatch_counts = {
    "ck_moe_stage1_fwd": 0, "ck_moe_stage2_fwd": 0,
    "hooked": False, "saw_oob_sentinel": False,
}
_LOG_DIR = "/tmp/ck_dispatch_logs"

def _patch_ck_dispatch():
    """Monkey-patch CK MoE stage1/stage2 to count invocations and detect OOB sentinels."""
    if _ck_dispatch_counts["hooked"]:
        return
    try:
        moe_op = sys.modules.get("aiter.ops.moe_op")
        if moe_op is None:
            return
        for fname in ("ck_moe_stage1_fwd", "ck_moe_stage2_fwd"):
            orig = getattr(moe_op, fname, None)
            if orig is None or getattr(orig, "_ck_hooked", False):
                continue
            @functools.wraps(orig)
            def _wrapped(*args, _orig=orig, _fname=fname, **kwargs):
                _ck_dispatch_counts[_fname] += 1
                # Check if sorted_token_ids contains OOB sentinel values
                # within the active dispatch range (num_valid_ids).
                # Signature: (hidden_states, w1, w2, sorted_token_ids,
                #             sorted_expert_ids, num_valid_ids, ...)
                if not _ck_dispatch_counts["saw_oob_sentinel"] and len(args) >= 6:
                    try:
                        hidden_states = args[0]
                        sorted_token_ids = args[3]
                        num_valid_ids = args[5]
                        n_valid = int(num_valid_ids[0].item()) if hasattr(num_valid_ids, '__getitem__') else int(num_valid_ids)
                        num_tokens = hidden_states.shape[0]
                        active_ids = sorted_token_ids[:n_valid]
                        max_id = active_ids.max().item()
                        if max_id >= num_tokens:
                            _ck_dispatch_counts["saw_oob_sentinel"] = True
                    except Exception:
                        pass
                return _orig(*args, **kwargs)
            _wrapped._ck_hooked = True
            setattr(moe_op, fname, _wrapped)
        _ck_dispatch_counts["hooked"] = True
    except Exception:
        pass

# Hook into the import system: after any import, check if aiter.ops.moe_op
# appeared and patch it. This catches both direct and transitive imports.
_orig_import = builtins.__import__
def _import_hook(name, *args, **kwargs):
    result = _orig_import(name, *args, **kwargs)
    if not _ck_dispatch_counts["hooked"] and "aiter.ops.moe_op" in sys.modules:
        _patch_ck_dispatch()
    return result
builtins.__import__ = _import_hook

def _write_log():
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        log_path = os.path.join(_LOG_DIR, f"{os.getpid()}.json")
        with open(log_path, "w") as f:
            json.dump(_ck_dispatch_counts, f)
    except Exception:
        pass

atexit.register(_write_log)
'''


def install_dispatch_hook():
    """Install the sitecustomize hook for the benchmark subprocess."""
    os.makedirs(HOOK_DIR, exist_ok=True)
    with open(os.path.join(HOOK_DIR, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE_SRC)
    # Clean any stale per-process logs
    shutil.rmtree(DISPATCH_LOG_DIR, ignore_errors=True)


def cleanup_dispatch_hook():
    """Remove the hook directory."""
    shutil.rmtree(HOOK_DIR, ignore_errors=True)


def read_dispatch_log():
    """Read and aggregate per-process dispatch logs. Returns dict or None."""
    if not os.path.isdir(DISPATCH_LOG_DIR):
        return None
    log_files = glob.glob(os.path.join(DISPATCH_LOG_DIR, "*.json"))
    if not log_files:
        return None
    aggregate = {
        "ck_moe_stage1_fwd": 0, "ck_moe_stage2_fwd": 0,
        "hooked": False, "saw_oob_sentinel": False, "num_workers": 0,
    }
    for lf in log_files:
        try:
            with open(lf, "r") as f:
                data = json.load(f)
            aggregate["ck_moe_stage1_fwd"] += data.get("ck_moe_stage1_fwd", 0)
            aggregate["ck_moe_stage2_fwd"] += data.get("ck_moe_stage2_fwd", 0)
            if data.get("hooked", False):
                aggregate["hooked"] = True
            if data.get("saw_oob_sentinel", False):
                aggregate["saw_oob_sentinel"] = True
            aggregate["num_workers"] += 1
        except Exception:
            continue
    return aggregate if aggregate["num_workers"] > 0 else None


def clear_ck_jit_cache():
    """Clear CK MoE JIT artifacts before the benchmark."""
    jit_dir = "/sgl-workspace/aiter/aiter/jit"
    for pattern in ["module_moe_ck2stages*", "module_moe_cktile2stages*"]:
        for path in glob.glob(os.path.join(jit_dir, "build", pattern)):
            shutil.rmtree(path, ignore_errors=True)
        for path in glob.glob(os.path.join(jit_dir, pattern + ".so")):
            os.remove(path)


def run_benchmark():
    """Run the benchmark script with dispatch instrumentation."""
    script = "/workspace/bench_bf16_moe.sh"
    if not os.path.isfile(script):
        return None, "Benchmark script not found"

    # Prepend hook dir to PYTHONPATH so sitecustomize.py is picked up.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = HOOK_DIR + (":" + existing if existing else "")

    try:
        result = subprocess.run(
            ["bash", script],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        full_output = result.stdout + "\n" + result.stderr
        return full_output, None
    except subprocess.TimeoutExpired:
        return None, "Benchmark timed out after 60 minutes"
    except Exception as e:
        return None, f"Benchmark execution error: {e}"


def extract_score(output):
    """Extract SCORE from benchmark output."""
    match = re.search(r"SCORE:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None


def main():
    print("=" * 60)
    print("BF16 MoE OOB Test Harness")
    print("=" * 60)

    # Check 1: Benchmark script exists
    if not os.path.isfile("/workspace/bench_bf16_moe.sh"):
        print("\nCHECK 1 [FAIL]: bench_bf16_moe.sh not found")
        print("SCORE: 0.0")
        return
    print("\nCHECK 1 [PASS]: bench_bf16_moe.sh exists")

    # ---- Instrument and run benchmark ----

    # Install dispatch hook and clear JIT cache before benchmark
    print("\nInstalling CK dispatch instrumentation...")
    install_dispatch_hook()
    clear_ck_jit_cache()

    # Check 2: Run benchmark
    print("Running benchmark (this may take 30+ minutes for JIT compilation)...")
    bench_start = time.time()
    output, error = run_benchmark()
    cleanup_dispatch_hook()

    if error:
        print(f"\nCHECK 2 [FAIL]: {error}")
        print("SCORE: 0.0")
        return
    print("CHECK 2 [PASS]: Benchmark completed")

    # Check 3: No GPU memory faults
    fault_patterns = [
        r"memory access fault",
        r"page fault",
        r"segfault",
        r"SIGSEGV",
        r"SIGBUS",
        r"core dump",
        r"GPU coredump",
    ]
    has_fault = any(re.search(p, output, re.IGNORECASE) for p in fault_patterns)
    if has_fault:
        print("CHECK 3 [FAIL]: GPU memory fault detected in output")
        print("SCORE: 0.0")
        return
    print("CHECK 3 [PASS]: No GPU memory faults")

    # Check 4: Valid score from benchmark
    score = extract_score(output)
    if score is None:
        print("CHECK 4 [FAIL]: Could not extract SCORE from output")
        print("SCORE: 0.0")
        return
    if score != 100.0:
        print(f"CHECK 4 [FAIL]: Benchmark score is {score}, expected 100.0")
        print(f"SCORE: {score}")
        return
    print(f"CHECK 4 [PASS]: Benchmark score: {score}")

    # Check 5: CK dispatch was exercised during the benchmark (runtime proof).
    # The injected sitecustomize hook patches ck_moe_stage1_fwd/ck_moe_stage2_fwd
    # and logs invocation counts.
    dispatch_log = read_dispatch_log()
    if dispatch_log is None:
        print("\nCHECK 5 [FAIL]: Dispatch logs not found.")
        print("  Could not verify that the CK MoE path was used at runtime.")
        print("SCORE: 0.0")
        return

    num_workers = dispatch_log.get("num_workers", 0)
    stage1_calls = dispatch_log.get("ck_moe_stage1_fwd", 0)
    stage2_calls = dispatch_log.get("ck_moe_stage2_fwd", 0)
    total_ck_calls = stage1_calls + stage2_calls
    was_hooked = dispatch_log.get("hooked", False)

    if not was_hooked:
        print("\nCHECK 5 [FAIL]: CK dispatch hook was not activated.")
        print(f"  {num_workers} worker log(s) found, but none recorded a hook activation.")
        print("  aiter.ops.moe_op was not imported during the benchmark.")
        print("SCORE: 0.0")
        return

    if total_ck_calls == 0:
        print("\nCHECK 5 [FAIL]: CK MoE dispatch functions were never called.")
        print(f"  stage1={stage1_calls}, stage2={stage2_calls} across {num_workers} worker(s)")
        print("  The benchmark must use the CK 2-stage MoE path, not another backend.")
        print("SCORE: 0.0")
        return

    print(f"CHECK 5 [PASS]: CK MoE dispatch verified "
          f"(stage1={stage1_calls}, stage2={stage2_calls}, "
          f"workers={num_workers})")

    # Check 6: OOB sentinel token IDs reached CK unchanged.
    # If consumer code sanitized sorted_token_ids before CK dispatch,
    # the wrapper would not observe any IDs >= num_tokens.
    saw_oob = dispatch_log.get("saw_oob_sentinel", False)
    if not saw_oob:
        print("\nCHECK 6 [FAIL]: No OOB sentinel token IDs observed at CK dispatch.")
        print("  sorted_token_ids never exceeded hidden_states.shape[0].")
        print("  Consumer code may be sanitizing sentinel IDs before CK dispatch.")
        print("SCORE: 0.0")
        return
    print("CHECK 6 [PASS]: OOB sentinel token IDs reached CK dispatch (no consumer sanitization)")

    # ---- Final result ----
    print("\n" + "=" * 60)
    print("All checks passed.")
    print("SCORE: 100.0")


if __name__ == "__main__":
    main()
