#!/usr/bin/env python3
"""Validate an eval instance end-to-end: schema check, Docker build,
preflight harness run, env probe, and write VERIFIED.md.

This is the "evals validate" command that ensures an instance can
actually reproduce a bug before launching an agent trial.

Usage:
    # Full validation (build + run + probe)
    python scripts/validate_instance.py evals/instances/sglang-fused-moe-fix/

    # Schema-only (no Docker, fast)
    python scripts/validate_instance.py --schema-only evals/instances/sglang-fused-moe-fix/

    # Skip build (use existing image)
    python scripts/validate_instance.py --skip-build evals/instances/sglang-fused-moe-fix/

    # All instances, schema only
    python scripts/validate_instance.py --all --schema-only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from instance_schema import validate_instance, ValidationResult

# Try to import env_probe (optional, for env checking)
try:
    from env_probe import probe_image, write_verified_md, update_metadata_verified, check_aiter_dependency
    HAS_ENV_PROBE = True
except ImportError:
    HAS_ENV_PROBE = False


def get_image_name(instance_dir: Path) -> str:
    """Get the Docker image name for an instance."""
    task_yaml = instance_dir / "task.yaml"
    if task_yaml.exists():
        content = task_yaml.read_text()
        m = re.search(r"base_image:\s*(\S+)", content)
        if m:
            return m.group(1)
    return f"amdpilot-eval-{instance_dir.name}"


def build_image(instance_dir: Path, image_name: str, timeout: int = 600) -> tuple[bool, str]:
    """Build Docker image from instance directory.

    Returns (success, output).
    """
    print(f"  Building Docker image '{image_name}'...")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "."],
            cwd=instance_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr[-500:]
        return True, result.stdout[-200:]
    except subprocess.TimeoutExpired:
        return False, f"Docker build timed out after {timeout}s"
    except FileNotFoundError:
        return False, "docker command not found"
    except Exception as e:
        return False, str(e)


def run_harness(image_name: str, instance_dir: Path, timeout: int = 300) -> tuple[float | None, str]:
    """Run test harness inside container and extract score.

    Returns (score_or_None, output).
    """
    # Read task.yaml for benchmark command
    task_yaml = instance_dir / "task.yaml"
    command = "/opt/venv/bin/python3 /workspace/test_harness.py"
    if task_yaml.exists():
        content = task_yaml.read_text()
        m = re.search(r"command:\s*[\"']?(.+?)[\"']?\s*$", content, re.MULTILINE)
        if m:
            command = m.group(1).strip().strip("\"'")

    # Read metric pattern
    metric_pattern = r"SCORE:\s+([\d.]+)"
    if task_yaml.exists():
        content = task_yaml.read_text()
        m = re.search(r"metric_pattern:\s*[\"'](.+?)[\"']\s*$", content, re.MULTILINE)
        if m:
            metric_pattern = m.group(1)

    container_name = f"amdpilot-validate-{instance_dir.name}-{int(time.time())}"
    print(f"  Running harness in container '{container_name}'...")
    print(f"  Command: {command}")

    try:
        # Run with GPU access
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--name", container_name,
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--shm-size=16g",
                "--group-add", "video",
                "--cap-add=SYS_PTRACE",
                "--security-opt", "seccomp=unconfined",
                image_name,
                "bash", "-c", command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = (result.stdout or "") + "\n" + (result.stderr or "")

        # Extract score
        m = re.search(metric_pattern, output)
        if m:
            score = float(m.group(1))
            return score, output[-500:]
        else:
            return None, f"No SCORE found in output. Last 500 chars:\n{output[-500:]}"

    except subprocess.TimeoutExpired:
        # Try cleanup
        subprocess.run(["docker", "rm", "-f", container_name],
                        capture_output=True, timeout=10)
        return None, f"Harness timed out after {timeout}s"
    except FileNotFoundError:
        return None, "docker command not found"
    except Exception as e:
        return None, str(e)


def validate_full(
    instance_dir: Path,
    skip_build: bool = False,
    schema_only: bool = False,
    build_timeout: int = 600,
    harness_timeout: int = 300,
) -> tuple[ValidationResult, dict]:
    """Full validation: schema → build → harness → env probe → VERIFIED.md

    Returns (validation_result, extra_info).
    """
    extra: dict = {}

    # Step 1: Schema validation
    print(f"\n{'='*60}")
    print(f"Validating: {instance_dir.name}")
    print(f"{'='*60}")

    print("\n[Step 1] Schema validation...")
    result = validate_instance(instance_dir)
    for e in result.errors:
        print(f"  ERROR: {e}")
    for w in result.warnings:
        print(f"  WARN:  {w}")
    if result.passed:
        print("  Schema: PASS")
    else:
        print("  Schema: FAIL")

    if schema_only:
        return result, extra

    if not result.passed:
        print("\n  Skipping Docker steps — schema validation failed.")
        return result, extra

    # Step 2: Docker build
    image_name = get_image_name(instance_dir)
    if skip_build:
        print(f"\n[Step 2] Docker build: SKIPPED (--skip-build, using '{image_name}')")
    else:
        print(f"\n[Step 2] Docker build...")
        success, build_output = build_image(instance_dir, image_name, build_timeout)
        if not success:
            result.error(f"Docker build failed: {build_output}")
            print(f"  Build: FAIL\n  {build_output[:300]}")
            return result, extra
        print("  Build: PASS")
        extra["build_success"] = True

    # Step 3: Run harness (preflight)
    print(f"\n[Step 3] Preflight harness run...")
    score, harness_output = run_harness(image_name, instance_dir, harness_timeout)

    # Determine instance category to interpret score semantics
    instance_category = None
    meta_path = instance_dir / "metadata.json"
    if meta_path.exists():
        try:
            _meta = json.loads(meta_path.read_text())
            instance_category = _meta.get("category")
        except Exception:
            pass

    if score is not None:
        extra["preflight_score"] = score
        print(f"  Preflight score: {score}")

        if instance_category == "optimize":
            # Optimize instances use performance metrics (throughput, latency),
            # not 0-100 bug-detection scores. Any non-None score means the
            # harness ran successfully and produced a baseline measurement.
            print(f"  Preflight: PASS (optimize baseline={score}, harness functional)")
        elif score >= 100.0:
            result.error(
                f"Preflight baseline score is {score} — bug does not reproduce. "
                "Check commit pin, harness, and model."
            )
            print("  Preflight: FAIL (bug not reproduced, score=100)")
        elif score <= 0.0:
            result.warn(f"Preflight score is {score} — harness may be too strict or crashing")
            print(f"  Preflight: WARN (score=0, check harness)")
        else:
            print(f"  Preflight: PASS (bug reproduces, score={score})")
    else:
        result.error(f"Harness failed to produce a score: {harness_output[:200]}")
        print(f"  Preflight: FAIL (no score)")
        extra["harness_error"] = harness_output[:500]

    # Step 4: Env probe
    if HAS_ENV_PROBE:
        print(f"\n[Step 4] Environment probe...")
        probe_result = probe_image(image_name)
        if "error" in probe_result:
            result.warn(f"Env probe failed: {probe_result['error']}")
            print(f"  Env probe: WARN ({probe_result['error']})")
        else:
            extra["env_probe"] = probe_result
            # Print key info
            for key, label in [
                ("aiter_version", "aiter"),
                ("sglang_version", "SGLang"),
                ("vllm_version", "vLLM"),
                ("rocm_version", "ROCm"),
                ("aiter_fused_mla", "fused_mla"),
            ]:
                val = probe_result.get(key)
                if val:
                    display = val.split("\n")[0][:50]
                    print(f"  {label}: {display}")

            # Check aiter dependency
            meta_path = instance_dir / "metadata.json"
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                    warnings = check_aiter_dependency(probe_result, metadata)
                    for w in warnings:
                        result.warn(w)
                        print(f"  WARN: {w}")
                except Exception:
                    pass

            print("  Env probe: DONE")

            # Step 5: Write VERIFIED.md
            print(f"\n[Step 5] Writing VERIFIED.md...")
            try:
                verified_path = write_verified_md(
                    instance_dir, probe_result,
                    preflight_score=score,
                    verified_by="validate_instance.py",
                )
                update_metadata_verified(
                    instance_dir, probe_result,
                    preflight_score=score,
                    verified_by="validate_instance.py",
                )
                print(f"  Wrote {verified_path}")
                print(f"  Updated {instance_dir / 'metadata.json'}")
            except Exception as e:
                result.warn(f"Failed to write VERIFIED.md: {e}")
    else:
        print(f"\n[Step 4] Env probe: SKIPPED (env_probe module not available)")

    return result, extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate eval instance(s) end-to-end"
    )
    parser.add_argument("instance_dirs", nargs="*",
                        help="Instance directory path(s) to validate")
    parser.add_argument("--all", action="store_true",
                        help="Validate all instances in evals/instances/")
    parser.add_argument("--schema-only", action="store_true",
                        help="Only run schema validation (no Docker)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker build (use existing image)")
    parser.add_argument("--build-timeout", type=int, default=600,
                        help="Docker build timeout in seconds (default: 600)")
    parser.add_argument("--harness-timeout", type=int, default=300,
                        help="Harness run timeout in seconds (default: 300)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    # Collect instance directories
    dirs: list[Path] = []
    if args.all:
        evals_dir = Path(__file__).resolve().parent.parent / "instances"
        dirs = sorted([d for d in evals_dir.iterdir() if d.is_dir()])
    elif args.instance_dirs:
        dirs = [Path(d).resolve() for d in args.instance_dirs]
    else:
        parser.error("Specify instance directory(ies) or --all")

    # Validate each
    results: list[dict] = []
    for d in dirs:
        if not d.is_dir():
            print(f"[SKIP] {d} — not a directory")
            continue

        result, extra = validate_full(
            d,
            skip_build=args.skip_build,
            schema_only=args.schema_only,
            build_timeout=args.build_timeout,
            harness_timeout=args.harness_timeout,
        )

        print(f"\n{'─'*40}")
        print(result.summary())

        results.append({
            "instance": d.name,
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            **extra,
        })

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        score_str = ""
        if "preflight_score" in r:
            score_str = f" (preflight={r['preflight_score']})"
        print(f"  [{status}] {r['instance']}{score_str}")
    print(f"\nTotal: {total}  Passed: {passed}  Failed: {failed}")

    if args.json:
        # Clean up for JSON output
        print(json.dumps(results, indent=2, default=str))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
