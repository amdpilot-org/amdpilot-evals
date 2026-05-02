#!/usr/bin/env python3
"""Synthetic harness wrapper for formulation-preview benchmarks."""

import re
import subprocess
import sys
from pathlib import Path

COMMAND = "bash /workspace/bench_mfu.sh"
METRIC_NAME = "TFLOPS_PER_GPU"
METRIC_PATTERN = re.compile("TFLOPS_PER_GPU:\\s+([\\d.]+)")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def main() -> int:
    result = subprocess.run(
        COMMAND,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/workspace",
    )
    output = (result.stdout or "") + (result.stderr or "")
    try:
        Path("/workspace/bench_output.log").write_text(output)
    except OSError:
        pass
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    match = METRIC_PATTERN.search(_strip_ansi(output))
    if match:
        try:
            metric_value = float(match.group(1))
        except (TypeError, ValueError):
            metric_value = None
        if metric_value is not None:
            print(f"{METRIC_NAME}: {metric_value}")
            print(f"SCORE: {metric_value}")
            return 0
    if result.returncode != 0:
        print(f"ERROR: benchmark exited with code {result.returncode}", file=sys.stderr)
        return result.returncode
    print(
        f"ERROR: failed to parse {METRIC_NAME} using pattern {METRIC_PATTERN.pattern}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
