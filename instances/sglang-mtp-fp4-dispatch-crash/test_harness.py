#!/usr/bin/env python3
"""
Test harness for sglang MTP crash with FP4/FP8 dispatch.

Validates that MTP layers function correctly when the main model uses
a different quantization format.
"""

import ast
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SGLANG_ROOT = "/workspace/sglang"
PYTHON_SRC = os.path.join(SGLANG_ROOT, "python", "sglang", "srt")

# Dynamically discover MoE-related source files
def _discover_moe_files():
    """Walk the source tree to find MoE dispatch and model files."""
    found = {}
    for root, dirs, files in os.walk(PYTHON_SRC):
        for f in files:
            if f.endswith('.py'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath) as fh:
                        content = fh.read()
                    # Include files that contain MoE dispatch logic
                    if any(kw in content for kw in ['moe', 'MoE', 'fused_moe', 'moriep', 'dispatch']):
                        found[f] = fpath
                except Exception:
                    pass
    return found

TARGET_FILES = _discover_moe_files()

# Model path for behavioral testing (volume-mounted from host)
MODEL_CACHE = "/root/.cache/huggingface/"

# Error pattern that indicates the bug is NOT fixed
CRASH_PATTERN = re.compile(r"Unsupported kernel config for moe heuristic dispatch")

# Server config
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 30000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    msg = f"[{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def read_source(path: str) -> str:
    """Read a source file, return empty string if missing."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def parse_ast(source: str, path: str):
    """Parse source to AST, return None on failure."""
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as e:
        print(f"  WARNING: SyntaxError parsing {path}: {e}")
        return None


class NameCollector(ast.NodeVisitor):
    """Collect all Name and Attribute nodes as strings."""

    def __init__(self):
        self.names = set()
        self.strings = set()

    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.names.add(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.strings.add(node.value)
        self.generic_visit(node)


def collect_names_and_strings(tree: ast.AST):
    """Return (set_of_names, set_of_string_literals) from an AST."""
    collector = NameCollector()
    collector.visit(tree)
    return collector.names, collector.strings


# ---------------------------------------------------------------------------
# AST-based source analysis tests
# ---------------------------------------------------------------------------


def test_bf16_weight_detection():
    """
    Check that the MoE dispatch code detects BF16 weights and applies
    a dequantization fallback rather than crashing.

    We look for references to bfloat16 / bf16 / dequant / fallback patterns
    in the relevant MoE layer files.
    """
    bf16_keywords = {"bfloat16", "bf16", "BFloat16"}
    dequant_keywords = {"dequant", "dequantize", "fallback", "to_float", "to_dtype"}

    files_to_check = list(TARGET_FILES.values())

    found_bf16_check = False
    found_dequant_logic = False
    details = []

    for fpath in files_to_check:
        source = read_source(fpath)
        if not source:
            continue

        tree = parse_ast(source, fpath)
        if tree is None:
            continue

        names, strings = collect_names_and_strings(tree)
        all_tokens = names | strings
        fname = os.path.basename(fpath)

        # Also do a raw text search for patterns that AST might miss
        source_lower = source.lower()

        if any(kw in all_tokens or kw.lower() in source_lower for kw in bf16_keywords):
            found_bf16_check = True
            details.append(f"{fname}: BF16 weight detection found")

        if any(
            kw in all_tokens or kw.lower() in source_lower for kw in dequant_keywords
        ):
            found_dequant_logic = True
            details.append(f"{fname}: dequant/fallback logic found")

    passed = found_bf16_check and found_dequant_logic
    record(
        "bf16_weight_detection_and_dequant_fallback",
        passed,
        "; ".join(details) if details else "No BF16 detection or dequant logic found",
    )
    return passed


def test_nextn_env_vars():
    """
    Check that independent env vars exist for controlling MTP/NextN layer
    dispatch separately from the main model dispatch.

    We look for env var references like SGLANG_NEXTN or similar patterns
    that allow per-layer dispatch configuration.
    """
    nextn_env_patterns = [
        r"SGLANG.*NEXTN",
        r"NEXTN.*DISPATCH",
        r"MTP.*DISPATCH",
        r"NEXT.*N.*MOE",
        r"nextn.*dispatch",
        r"mtp.*dispatch",
    ]

    files_to_check = list(TARGET_FILES.values())
    found_env_var = False
    details = []

    for fpath in files_to_check:
        source = read_source(fpath)
        if not source:
            continue

        fname = os.path.basename(fpath)

        for pattern in nextn_env_patterns:
            matches = re.findall(pattern, source, re.IGNORECASE)
            if matches:
                found_env_var = True
                details.append(f"{fname}: env var pattern '{matches[0]}' found")

    # Also check for os.environ / os.getenv calls with nextn-related strings
    for fpath in files_to_check:
        source = read_source(fpath)
        if not source:
            continue

        tree = parse_ast(source, fpath)
        if tree is None:
            continue

        _, strings = collect_names_and_strings(tree)
        fname = os.path.basename(fpath)

        for s in strings:
            s_upper = s.upper()
            if ("NEXTN" in s_upper or "NEXT_N" in s_upper) and (
                "DISPATCH" in s_upper or "MOE" in s_upper
            ):
                found_env_var = True
                details.append(f"{fname}: env var string '{s}' found")

    record(
        "nextn_dispatch_env_vars",
        found_env_var,
        "; ".join(details) if details else "No NEXTN dispatch env vars found",
    )
    return found_env_var


def test_is_nextn_flag_propagation():
    """
    Check that an is_nextn flag (or equivalent) is propagated through the
    MoE layer stack so that dispatch logic can differentiate MTP layers
    from main model layers.
    """
    flag_patterns = [
        r"is_nextn",
        r"is_next_n",
        r"is_mtp",
        r"nextn_layer",
        r"mtp_layer",
    ]

    files_to_check = list(TARGET_FILES.values())
    found_in_files = set()
    details = []

    for fpath in files_to_check:
        source = read_source(fpath)
        if not source:
            continue

        fname = os.path.basename(fpath)

        for pattern in flag_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                found_in_files.add(fname)
                details.append(f"{fname}: flag pattern '{pattern}' found")

    # The flag should be propagated across multiple files (at least 2)
    passed = len(found_in_files) >= 2
    record(
        "is_nextn_flag_propagation",
        passed,
        "; ".join(details)
        if details
        else "is_nextn flag not propagated across enough files",
    )
    return passed


def test_no_crash_pattern_in_moe_dispatch():
    """
    Check that the code path that raises 'Unsupported kernel config for moe
    heuristic dispatch' now has proper handling for the BF16+FP4/FP8 case
    so it won't be reached when MTP layers have BF16 weights.

    We look for guard conditions before the error raise, or alternative
    dispatch paths that handle this dtype combination.
    """
    files_to_check = list(TARGET_FILES.values())

    found_error_site = False
    has_guard = False
    details = []

    for fpath in files_to_check:
        source = read_source(fpath)
        if not source:
            continue

        fname = os.path.basename(fpath)

        # Find the error message location
        if "Unsupported kernel config for moe heuristic dispatch" in source:
            found_error_site = True
            details.append(f"{fname}: error raise site found")

            # Check if there's a guard / alternative path before the error.
            # Look for dtype checks, bfloat16 conditionals, or fallback logic
            # near the error site.
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if "Unsupported kernel config for moe heuristic dispatch" in line:
                    # Check surrounding context (20 lines before)
                    context_start = max(0, i - 20)
                    context = "\n".join(lines[context_start:i])
                    context_lower = context.lower()

                    if any(
                        kw in context_lower
                        for kw in [
                            "bfloat16",
                            "bf16",
                            "dequant",
                            "fallback",
                            "is_nextn",
                            "is_next_n",
                            "is_mtp",
                        ]
                    ):
                        has_guard = True
                        details.append(
                            f"{fname}: guard/fallback found near error site"
                        )

        # Also check for new dispatch paths that handle mixed dtype
        source_lower = source.lower()
        if "fp4" in source_lower or "fp8" in source_lower:
            if "bf16" in source_lower or "bfloat16" in source_lower:
                has_guard = True
                details.append(f"{fname}: mixed FP4/FP8 + BF16 handling found")

    # If the error site is gone entirely, that's also acceptable
    if not found_error_site:
        has_guard = True
        details.append("Error raise site removed or refactored away")

    record(
        "crash_path_guarded",
        has_guard,
        "; ".join(details) if details else "No guard found for crash path",
    )
    return has_guard


def test_deepseek_v2_nextn_support():
    """
    Check that deepseek_v2.py propagates the is_nextn flag or equivalent
    so that MTP layers get distinct dispatch treatment.
    """
    # Find the DeepseekV2 model file dynamically
    fpath = None
    for name, path in TARGET_FILES.items():
        if 'deepseek' in name.lower():
            fpath = path
            break
    if fpath is None:
        # Search for it
        for root, dirs, files in os.walk(os.path.join(PYTHON_SRC, "models")):
            for f in files:
                if 'deepseek' in f.lower() and f.endswith('.py'):
                    fpath = os.path.join(root, f)
                    break

    if fpath is None:
        record("deepseek_v2_nextn_support", False, "DeepSeek model file not found")
        return False

    source = read_source(fpath)
    if not source:
        record("deepseek_v2_nextn_support", False, "DeepSeek model file empty")
        return False

    flag_patterns = [
        r"is_nextn",
        r"is_next_n",
        r"is_mtp",
        r"nextn",
    ]

    found = []
    for pattern in flag_patterns:
        matches = re.findall(pattern, source, re.IGNORECASE)
        if matches:
            found.extend(matches)

    passed = len(found) > 0
    record(
        "deepseek_v2_nextn_support",
        passed,
        f"Found references: {', '.join(set(found))}"
        if found
        else "No nextn/mtp flag references in deepseek_v2.py",
    )
    return passed


# ---------------------------------------------------------------------------
# Behavioral tests (only if model is available)
# ---------------------------------------------------------------------------


def find_mxfp4_model():
    """
    Look for an MXFP4 DeepSeek model in the HuggingFace cache.
    Returns model path if found, None otherwise.
    """
    if not os.path.isdir(MODEL_CACHE):
        return None

    # Look for common MXFP4 DeepSeek model directories
    candidate_patterns = [
        "deepseek",
        "DeepSeek",
    ]

    for root, dirs, files in os.walk(MODEL_CACHE):
        # Don't recurse too deep
        depth = root.replace(MODEL_CACHE, "").count(os.sep)
        if depth > 3:
            continue

        for d in dirs:
            d_lower = d.lower()
            if any(p.lower() in d_lower for p in candidate_patterns):
                full_path = os.path.join(root, d)
                # Check for config.json to verify it's a model dir
                config_path = os.path.join(full_path, "config.json")
                if os.path.isfile(config_path):
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        quant_config = config.get("quantization_config", {})
                        if quant_config.get("quant_method") in [
                            "fp4",
                            "mxfp4",
                            "fbgemm_fp8",
                        ]:
                            return full_path
                    except (json.JSONDecodeError, KeyError):
                        pass

    return None


def start_server(model_path: str):
    """
    Start sglang server with the given model, EAGLE spec decode, and MORI dispatch.
    Returns the subprocess.Popen object.
    """
    env = os.environ.copy()
    env["SGLANG_MOE_DISPATCH"] = "mori"

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(SERVER_PORT),
        "--host",
        SERVER_HOST,
        "--tp",
        "2",
        "--trust-remote-code",
        "--speculative-algorithm",
        "EAGLE",
    ]

    print(f"Starting server with command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc


def wait_for_server(proc, timeout=300):
    """
    Wait for server to start up. Returns (success, logs).
    - success=True: server is ready
    - success=False: server crashed or timed out
    """
    logs = []
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if process has exited (crashed)
        ret = proc.poll()
        if ret is not None:
            # Process exited - read remaining output
            remaining = proc.stdout.read()
            if remaining:
                logs.append(remaining)
            return False, "".join(logs)

        # Try to read available output (non-blocking)
        try:
            line = proc.stdout.readline()
            if line:
                logs.append(line)
                print(f"  SERVER: {line.rstrip()}")

                # Check for crash pattern
                if CRASH_PATTERN.search(line):
                    return False, "".join(logs)
        except Exception:
            pass

        # Check if server is ready
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health")
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                return True, "".join(logs)
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass

        time.sleep(2)

    return False, "".join(logs)


def send_inference_request():
    """Send a simple inference request to verify the server works."""
    payload = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        return True, content
    except Exception as e:
        return False, str(e)


def kill_server(proc):
    """Kill the server process and its children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_behavioral_tests():
    """Run behavioral tests if an MXFP4 model is available."""
    model_path = find_mxfp4_model()

    if model_path is None:
        print("\n--- Behavioral Tests: SKIPPED (no MXFP4 model found) ---")
        print(f"  Looked in: {MODEL_CACHE}")
        print("  To run behavioral tests, mount an MXFP4 model to the container.")
        # Model absent — behavioral verification cannot proceed, record as FAIL
        record(
            "behavioral_server_startup",
            False,
            "No MXFP4 model available — behavioral verification required",
        )
        record(
            "behavioral_inference",
            False,
            "No MXFP4 model available — cannot verify fix at runtime",
        )
        return

    print(f"\n--- Behavioral Tests: using model {model_path} ---")

    proc = start_server(model_path)
    try:
        server_ready, logs = wait_for_server(proc, timeout=300)

        # Check for the crash pattern in logs
        crash_found = bool(CRASH_PATTERN.search(logs))

        if crash_found:
            record(
                "behavioral_server_startup",
                False,
                "Server crashed with 'Unsupported kernel config for moe heuristic dispatch'",
            )
            record(
                "behavioral_inference",
                False,
                "Server did not start - crash during CUDA graph capture",
            )
            return

        if not server_ready:
            record(
                "behavioral_server_startup",
                False,
                "Server did not become ready within timeout (no crash pattern found, "
                "but server is not healthy)",
            )
            record(
                "behavioral_inference",
                False,
                "Server not ready",
            )
            return

        record("behavioral_server_startup", True, "Server started without crash")

        # Send inference request
        success, response = send_inference_request()
        record(
            "behavioral_inference",
            success,
            f"Response: {response[:200]}" if success else f"Error: {response}",
        )
    finally:
        kill_server(proc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Test Harness: sglang-mtp-fp4-dispatch-crash")
    print("=" * 70)

    # --- AST-based source analysis ---
    print("\n--- AST-based Source Analysis ---")

    print(f"  Discovered {len(TARGET_FILES)} MoE-related source files")

    test_bf16_weight_detection()
    test_nextn_env_vars()
    test_is_nextn_flag_propagation()
    test_no_crash_pattern_in_moe_dispatch()
    test_deepseek_v2_nextn_support()

    # --- Behavioral tests ---
    run_behavioral_tests()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    for name, p, detail in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")

    print(f"\nTotal: {total}  Passed: {passed}  Failed: {failed}")

    score = (passed / total * 100.0) if total > 0 else 0.0
    print(f"\nResults: {passed}/{total}")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
