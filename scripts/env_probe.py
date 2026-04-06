#!/usr/bin/env python3
"""Container environment probe for amdpilot-evals validation.

Probes a Docker container (running or via one-shot exec) for software
environment info: aiter/sglang/vllm versions, ROCm version, Python version,
PYTHONPATH, and aiter fused MLA availability.

Usage:
    # Probe a running container
    python scripts/env_probe.py --container <container_name>

    # Probe a Docker image (starts temp container, probes, removes)
    python scripts/env_probe.py --image <image_name>

    # Output as JSON (for piping into validate)
    python scripts/env_probe.py --container <name> --json

    # Write to VERIFIED.md
    python scripts/env_probe.py --container <name> --write-verified <instance_dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE_SCRIPT = r"""
echo "AITER_VERSION_START"
cd /sgl-workspace/aiter 2>/dev/null && git log --oneline -1 2>/dev/null && git describe --tags --always 2>/dev/null || echo "AITER_NOT_FOUND"
echo "AITER_VERSION_END"
echo "SGLANG_VERSION_START"
cd /sgl-workspace/sglang 2>/dev/null && git log --oneline -1 2>/dev/null && git describe --tags --always 2>/dev/null || echo "SGLANG_NOT_FOUND"
echo "SGLANG_VERSION_END"
echo "VLLM_VERSION_START"
cd /workspace/vllm 2>/dev/null && git log --oneline -1 2>/dev/null && git describe --tags --always 2>/dev/null || echo "VLLM_NOT_FOUND"
echo "VLLM_VERSION_END"
echo "ROCM_VERSION_START"
cat /opt/rocm/.info/version 2>/dev/null || echo "ROCM_NOT_FOUND"
echo "ROCM_VERSION_END"
echo "PYTHON_VERSION_START"
python3 --version 2>/dev/null || echo "PYTHON_NOT_FOUND"
echo "PYTHON_VERSION_END"
echo "PYTHONPATH_START"
echo "${PYTHONPATH:-NOT_SET}"
echo "PYTHONPATH_END"
echo "AITER_FUSED_MLA_START"
python3 -c "from aiter import fused_mla; print('available')" 2>/dev/null || echo "not_available"
echo "AITER_FUSED_MLA_END"
echo "AITER_FUSED_MLA_ROPE_START"
python3 -c "from aiter import fused_mla_rope; print('available')" 2>/dev/null || echo "not_available"
echo "AITER_FUSED_MLA_ROPE_END"
""".strip()


def parse_probe_output(output: str) -> dict[str, Any]:
    """Parse output from the container env probe script."""
    result: dict[str, Any] = {}
    section = ""
    section_lines: list[str] = []

    for line in output.strip().splitlines():
        if line.endswith("_START"):
            section = line.rsplit("_START", 1)[0].lower()
            section_lines = []
        elif line.endswith("_END"):
            key = line.rsplit("_END", 1)[0].lower()
            if key == section and section_lines:
                content = "\n".join(section_lines).strip()
                if content.endswith("_NOT_FOUND"):
                    result[section] = None
                else:
                    result[section] = content
            section = ""
            section_lines = []
        elif section:
            section_lines.append(line)

    return result


def probe_container(container_name: str, timeout: int = 30) -> dict[str, Any]:
    """Probe a running Docker container for software environment info."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "bash", "-c", PROBE_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"error": f"docker exec failed: {result.stderr[:200]}"}
        parsed = parse_probe_output(result.stdout)
        parsed["container_name"] = container_name
        return parsed
    except subprocess.TimeoutExpired:
        return {"error": f"probe timed out after {timeout}s"}
    except FileNotFoundError:
        return {"error": "docker command not found"}
    except Exception as e:
        return {"error": str(e)}


def probe_image(image_name: str, timeout: int = 60) -> dict[str, Any]:
    """Probe a Docker image by starting a temporary container."""
    container_name = f"amdpilot-env-probe-{int(datetime.now().timestamp())}"
    try:
        # Start temp container
        result = subprocess.run(
            ["docker", "run", "--rm", "--name", container_name,
             "--entrypoint", "bash", image_name, "-c", PROBE_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"error": f"docker run failed: {result.stderr[:200]}"}
        parsed = parse_probe_output(result.stdout)
        parsed["image_name"] = image_name
        return parsed
    except subprocess.TimeoutExpired:
        # Try to clean up
        subprocess.run(["docker", "rm", "-f", container_name],
                        capture_output=True, timeout=5)
        return {"error": f"probe timed out after {timeout}s"}
    except FileNotFoundError:
        return {"error": "docker command not found"}
    except Exception as e:
        return {"error": str(e)}


def check_aiter_dependency(probe_result: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Check if env probe results are compatible with instance metadata.

    Returns a list of warnings (empty if all OK).
    """
    warnings: list[str] = []

    if metadata.get("aiter_dependency"):
        if probe_result.get("aiter_fused_mla") == "not_available":
            warnings.append(
                "Instance requires aiter fused_mla but it is NOT available in container. "
                "Bug path cannot be triggered."
            )
        if probe_result.get("aiter_fused_mla_rope") == "not_available":
            warnings.append(
                "Instance requires aiter fused_mla_rope but it is NOT available in container. "
                "Bug path cannot be triggered."
            )

    if probe_result.get("aiter_version") is None and metadata.get("bug_source") == "aiter":
        warnings.append("Instance targets aiter bug but aiter is not installed in container.")

    if probe_result.get("sglang_version") is None and metadata.get("bug_source") == "sglang":
        warnings.append("Instance targets sglang bug but sglang is not installed in container.")

    if probe_result.get("vllm_version") is None and metadata.get("bug_source") == "vllm":
        warnings.append("Instance targets vllm bug but vllm is not installed in container.")

    return warnings


def write_verified_md(
    instance_dir: Path,
    probe_result: dict[str, Any],
    preflight_score: float | None = None,
    verified_by: str = "env_probe.py",
) -> Path:
    """Write VERIFIED.md to an instance directory."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Verification Record",
        "",
        f"**Date**: {now}",
        f"**Verified by**: {verified_by}",
    ]
    if preflight_score is not None:
        lines.append(f"**Preflight score**: {preflight_score}")

    lines.extend([
        "",
        "## Container Environment",
        "",
        f"| Component | Version |",
        f"|-----------|---------|",
    ])

    for key, label in [
        ("aiter_version", "aiter"),
        ("sglang_version", "SGLang"),
        ("vllm_version", "vLLM"),
        ("rocm_version", "ROCm"),
        ("python_version", "Python"),
    ]:
        val = probe_result.get(key)
        # Take first line only for display
        if val:
            display = val.split("\n")[0][:60]
        else:
            display = "N/A"
        lines.append(f"| {label} | `{display}` |")

    lines.extend([
        "",
        "## Capabilities",
        "",
        f"- **fused_mla**: {probe_result.get('aiter_fused_mla', 'unknown')}",
        f"- **fused_mla_rope**: {probe_result.get('aiter_fused_mla_rope', 'unknown')}",
        f"- **PYTHONPATH**: `{probe_result.get('pythonpath', 'unknown')}`",
        "",
    ])

    out_path = instance_dir / "VERIFIED.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def update_metadata_verified(
    instance_dir: Path,
    probe_result: dict[str, Any],
    preflight_score: float | None = None,
    verified_by: str = "env_probe.py",
) -> None:
    """Update metadata.json with verification info."""
    meta_path = instance_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata["verified"] = {
        "date": now,
        "by": verified_by,
        "preflight_score": preflight_score,
        "env": {
            "aiter_version": (probe_result.get("aiter_version") or "").split("\n")[0][:60] or None,
            "sglang_version": (probe_result.get("sglang_version") or "").split("\n")[0][:60] or None,
            "vllm_version": (probe_result.get("vllm_version") or "").split("\n")[0][:60] or None,
            "rocm_version": probe_result.get("rocm_version"),
            "aiter_fused_mla": probe_result.get("aiter_fused_mla"),
            "aiter_fused_mla_rope": probe_result.get("aiter_fused_mla_rope"),
        },
    }

    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Docker container/image environment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--container", help="Name of running Docker container to probe")
    group.add_argument("--image", help="Docker image to probe (starts temp container)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--write-verified", metavar="INSTANCE_DIR",
                        help="Write VERIFIED.md and update metadata.json in instance dir")
    parser.add_argument("--preflight-score", type=float, help="Preflight score to record")
    parser.add_argument("--check-metadata", metavar="METADATA_JSON",
                        help="Check env compatibility with metadata.json")
    args = parser.parse_args()

    # Probe
    if args.container:
        result = probe_container(args.container)
    else:
        result = probe_image(args.image)

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Container Environment Probe")
        print("=" * 40)
        for key, label in [
            ("aiter_version", "aiter"),
            ("sglang_version", "SGLang"),
            ("vllm_version", "vLLM"),
            ("rocm_version", "ROCm"),
            ("python_version", "Python"),
            ("pythonpath", "PYTHONPATH"),
            ("aiter_fused_mla", "fused_mla"),
            ("aiter_fused_mla_rope", "fused_mla_rope"),
        ]:
            val = result.get(key)
            if val:
                display = val.split("\n")[0][:60]
            else:
                display = "N/A"
            print(f"  {label:20s}: {display}")

    # Check metadata compatibility
    if args.check_metadata:
        meta = json.loads(Path(args.check_metadata).read_text())
        warnings = check_aiter_dependency(result, meta)
        if warnings:
            print("\nWARNINGS:", file=sys.stderr)
            for w in warnings:
                print(f"  ⚠️  {w}", file=sys.stderr)
            sys.exit(2)

    # Write verification record
    if args.write_verified:
        instance_dir = Path(args.write_verified)
        if not instance_dir.is_dir():
            print(f"ERROR: {instance_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        verified_path = write_verified_md(instance_dir, result, args.preflight_score)
        update_metadata_verified(instance_dir, result, args.preflight_score)
        print(f"\nWrote {verified_path}")
        print(f"Updated {instance_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
