#!/usr/bin/env python3
"""Test harness for vllm#35569: ROCM_ATTN backend score deviation on Qwen3-VL-Reranker pooling.

Formulation v2 fixes (from clean2 experiment learnings):
- Use --attention-backend CLI flag instead of VLLM_ATTENTION_BACKEND env var
- Increase wait_for_server timeout to 900s for MI355X cold model load
- Add --enforce-eager to reduce cold start time (skip torch.compile)
- NOTE: The scoring endpoint (/v1/score) is NOT pre-enabled — enabling it is
  part of the agent's challenge (requires patching enable_scoring_api()).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

PYTHON = "/usr/bin/python3"
MODEL = "Qwen/Qwen3-VL-Reranker-2B"
HOST = "127.0.0.1"
PORT = 18999
EXPECTED_SCORE = 0.100404
REL_TOL = 0.05  # 5% relative tolerance — should be achievable if fix works

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def wait_for_server(base_url: str, timeout: int = 900):
    """Wait for the vLLM server to be ready.

    Timeout increased to 900s for MI355X cold model load (~600-700s).
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"Server ready after {time.time() - start:.1f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"Server did not become ready within {timeout}s")
    return False


def score_request(base_url: str, text_1: str, text_2: str):
    """Send a score request to the vLLM server."""
    payload = json.dumps({
        "model": MODEL,
        "text_1": text_1,
        "text_2": text_2,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/score",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    base_url = f"http://{HOST}:{PORT}/v1"
    server_proc = None

    try:
        # Start vLLM server — use CLI flag for attention backend selection
        # (VLLM_ATTENTION_BACKEND env var is unrecognized in this build)
        env = os.environ.copy()
        env["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
        env["TOKENIZERS_PARALLELISM"] = "false"

        cmd = [
            PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--host", HOST,
            "--port", str(PORT),
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--max-num-seqs", "1",
            "--dtype", "auto",
            "--attention-backend", "ROCM_ATTN",
            "--enforce-eager",
        ]

        print(f"Starting vLLM server with ROCM_ATTN backend...")
        print(f"Command: {' '.join(cmd)}")
        server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Wait for server (900s timeout for MI355X cold model load)
        server_ready = wait_for_server(base_url, timeout=900)
        check("Server started successfully", server_ready, "Server did not start within timeout")

        if not server_ready:
            # Try to capture server output for debugging
            if server_proc.poll() is not None:
                stdout = server_proc.stdout.read().decode("utf-8", errors="replace")
                print(f"Server output (last 2000 chars):\n{stdout[-2000:]}")
            score = int(100 * checks_passed / checks_total) if checks_total > 0 else 0
            print(f"\nSCORE: {score}")
            sys.exit(1)

        # Test 1: text_vs_text score accuracy
        print("\n--- Test 1: text_vs_text score ---")
        text_1 = "What is the capital of France?"
        text_2 = "The capital of France is Paris."
        try:
            result = score_request(base_url, text_1, text_2)
            actual_score = result["data"][0]["score"]
            rel_diff = abs(actual_score - EXPECTED_SCORE) / abs(EXPECTED_SCORE)
            print(f"  [ROCM_ATTN] text_vs_text: actual={actual_score:.6f} expected={EXPECTED_SCORE:.6f} "
                  f"diff={abs(actual_score - EXPECTED_SCORE):.6f} rel_diff={rel_diff:.4f}")
            check(
                f"text_vs_text score within {REL_TOL*100}% tolerance",
                rel_diff <= REL_TOL,
                f"rel_diff={rel_diff:.4f} ({rel_diff*100:.2f}%) exceeds {REL_TOL*100}%"
            )
        except Exception as e:
            check("text_vs_text score request", False, str(e))

        # Test 2: Consistency check — run same query again to verify determinism
        print("\n--- Test 2: Score determinism ---")
        try:
            result2 = score_request(base_url, text_1, text_2)
            actual_score2 = result2["data"][0]["score"]
            score_diff = abs(actual_score2 - actual_score)
            print(f"  Run 2: actual={actual_score2:.6f}, diff from run 1={score_diff:.8f}")
            check(
                "Deterministic scores across runs",
                score_diff < 0.001,
                f"Scores differ by {score_diff:.6f}"
            )
        except Exception as e:
            check("Determinism check", False, str(e))

        # Test 3: Different query pair to check it's not just one input
        print("\n--- Test 3: Alternative query pair ---")
        text_1b = "How does photosynthesis work?"
        text_2b = "Photosynthesis converts sunlight into chemical energy in plants."
        try:
            result3 = score_request(base_url, text_1b, text_2b)
            actual_score3 = result3["data"][0]["score"]
            print(f"  Alternative query score: {actual_score3:.6f}")
            # Just check we get a valid score (non-zero, between -1 and 1 range roughly)
            check(
                "Alternative query returns valid score",
                actual_score3 is not None and isinstance(actual_score3, (int, float)),
                f"Got invalid score: {actual_score3}"
            )
        except Exception as e:
            check("Alternative query score request", False, str(e))

    finally:
        # Clean up server
        if server_proc is not None:
            print("\nShutting down server...")
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                server_proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    score = int(100 * checks_passed / checks_total) if checks_total > 0 else 0
    print(f"\nSCORE: {score}")
    sys.exit(0 if score == 100 else 1)


if __name__ == "__main__":
    main()
