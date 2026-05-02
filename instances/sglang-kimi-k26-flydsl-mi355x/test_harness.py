#!/usr/bin/env python3
"""Phase G accuracy harness for sglang-kimi-k26-flydsl-mi355x.

Runs ``lm_eval`` GSM8K (10-shot, limit=50) against a local SGLang server
serving Kimi-K2.6. If no server is listening on :30000, the harness
launches one with the PR #23381 canonical flags plus any overrides from
``/workspace/bench_config.env`` (same contract as ``bench_flydsl_k26.sh``).

Exits 0 with ``SCORE: 100.0`` iff ``exact_match,flexible-extract`` is
>= 0.90 (blog reports 0.96 at limit=100 for baseline and optimized). Exits
non-zero with ``SCORE: 0.0`` for any failure: server unhealthy, lm_eval
crashed, unparseable output, or accuracy below threshold.

Immutable per ``baseline_contract.immutable_artifacts`` in task.yaml;
amdpilot's harness integrity check restores this file if the agent
accidentally edits it.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"
PORT = 30000
HEALTH_URL = f"http://{HOST}:{PORT}/health"
MODEL_PATH = "/sgl-workspace/models/Kimi-K2.6"
SERVED_NAME = "Kimi-K2.6"
BENCH_CONFIG = Path("/workspace/bench_config.env")

LM_EVAL_BIN = "/opt/venv/bin/lm_eval"
LM_EVAL_LIMIT = 50
LM_EVAL_FEWSHOT = 10
THRESHOLD = 0.90

STARTUP_TIMEOUT_S = 900
LM_EVAL_TIMEOUT_S = 1500

SCORE_ZERO = "\nSCORE: 0.0"
SCORE_FULL = "\nSCORE: 100.0"


def log(msg: str) -> None:
    print(f"[harness] {msg}", flush=True)


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
            r.read(16)
        return True
    except Exception:
        return False


def source_bench_config(env: dict[str, str]) -> dict[str, str]:
    """Inherit any overrides the agent wrote to /workspace/bench_config.env."""
    if not BENCH_CONFIG.exists():
        return env
    try:
        out = subprocess.run(
            ["bash", "-c", f"set -a; source {BENCH_CONFIG}; env"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception as exc:
        log(f"WARN: could not source {BENCH_CONFIG}: {exc}")
        return env
    for line in out.stdout.splitlines():
        if "=" in line and not line.startswith(("_=", "SHLVL=", "PWD=")):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def launch_server() -> subprocess.Popen[bytes]:
    env = source_bench_config(os.environ.copy())
    extra = env.get("EXTRA_SERVER_FLAGS", "").split()
    cmd = [
        "/opt/venv/bin/python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--served-model-name",
        SERVED_NAME,
        "--tensor-parallel-size",
        "4",
        "--trust-remote-code",
        "--decode-attention-backend",
        "triton",
        "--prefill-attention-backend",
        "aiter",
        "--disable-custom-all-reduce",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "128000",
        "--skip-server-warmup",
        "--reasoning-parser",
        "kimi_k2",
        "--tool-call-parser",
        "kimi_k2",
        "--watchdog-timeout",
        "1200",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ] + extra
    log_path = Path("/tmp/test_harness_server.log")
    log(f"launching server (log: {log_path})")
    f = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


def wait_for_health(proc: subprocess.Popen[bytes] | None, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if health_ok():
            return True
        if proc is not None and proc.poll() is not None:
            log(f"server exited early with code {proc.returncode}")
            return False
        time.sleep(5)
    return False


def run_lm_eval() -> tuple[bool, float, str]:
    if not Path(LM_EVAL_BIN).exists():
        return False, 0.0, (
            f"{LM_EVAL_BIN} not found; install with "
            "`/opt/venv/bin/pip install lm-eval` (or bake into image)"
        )
    cmd = [
        LM_EVAL_BIN,
        "--model",
        "local-completions",
        "--model_args",
        (
            f"model={MODEL_PATH},"
            f"base_url=http://{HOST}:{PORT}/v1/completions,"
            "num_concurrent=1,tokenized_requests=False,trust_remote_code=True"
        ),
        "--tasks",
        "gsm8k",
        "--num_fewshot",
        str(LM_EVAL_FEWSHOT),
        "--limit",
        str(LM_EVAL_LIMIT),
    ]
    log(f"running lm_eval --limit {LM_EVAL_LIMIT} --num_fewshot {LM_EVAL_FEWSHOT}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=LM_EVAL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, 0.0, f"lm_eval timed out after {LM_EVAL_TIMEOUT_S}s"
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    print(output)
    if result.returncode != 0:
        return False, 0.0, f"lm_eval exited with code {result.returncode}"

    # Parse the gsm8k row's exact_match,flexible-extract value.
    # lm_eval emits a markdown table like:
    #   |Tasks|Version|Filter|n-shot|Metric|   |Value |   |Stderr|
    #   |-----|------:|------|-----:|------|---|-----:|---|------|
    #   |gsm8k|      3|flexible-extract|    10|exact_match|↑|0.9600|±  |0.0197|
    # Also tolerate JSON-style output from newer versions.
    patterns = [
        r"gsm8k[^\n]*flexible-extract[^\n]*exact_match[^\d\n]*([01]?\.\d+)",
        r'"exact_match,flexible-extract"\s*:\s*([01]?\.\d+)',
        r"exact_match,flexible-extract[^0-9\n]*([01]?\.\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            try:
                return True, float(m.group(1)), ""
            except ValueError:
                continue
    return False, 0.0, "could not parse exact_match,flexible-extract from lm_eval output"


def main() -> int:
    log("Phase G accuracy harness for sglang-kimi-k26-flydsl-mi355x")
    launched_by_us = False
    proc: subprocess.Popen[bytes] | None = None
    try:
        if health_ok():
            log(f"reusing server already healthy on :{PORT}")
        else:
            proc = launch_server()
            launched_by_us = True
            if not wait_for_health(proc, STARTUP_TIMEOUT_S):
                log(f"server did not become healthy within {STARTUP_TIMEOUT_S}s")
                print(SCORE_ZERO)
                return 1

        ok, metric, err = run_lm_eval()
        if not ok:
            log(f"lm_eval failed: {err}")
            print(SCORE_ZERO)
            return 1

        print(f"GSM8K_EXACT_MATCH_FLEXIBLE: {metric:.4f}")
        log(f"metric={metric:.4f} threshold={THRESHOLD:.2f}")
        if metric >= THRESHOLD:
            print(SCORE_FULL)
            return 0
        log(f"metric {metric:.4f} < threshold {THRESHOLD:.2f}")
        print(SCORE_ZERO)
        return 1
    finally:
        if launched_by_us and proc is not None:
            log("shutting down harness-launched server")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
            except Exception as exc:
                log(f"cleanup warning: {exc}")


if __name__ == "__main__":
    sys.exit(main())
