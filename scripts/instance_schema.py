#!/usr/bin/env python3
"""Canonical instance schema for amdpilot-evals.

Defines the expected structure and validates existing instances against it.
Every eval instance must conform to this schema before launch.

Usage:
    # Validate a single instance
    python scripts/instance_schema.py evals/instances/sglang-fused-moe-fix/

    # Validate all instances
    python scripts/instance_schema.py --all

    # Dump the schema as JSON
    python scripts/instance_schema.py --dump-schema
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Required files in every instance directory
REQUIRED_FILES = ["Dockerfile", "task.yaml", "test_harness.py", "task_description.md", "metadata.json"]

# Optional files
OPTIONAL_FILES = ["VERIFIED.md"]

# task.yaml required fields
TASK_YAML_REQUIRED = {
    "name": str,
    "type": str,           # bugfix | optimize | feature
    "repo": str,           # https://github.com/org/repo.git
    "base_image": str,
    "benchmark": dict,     # must have command, metric_name, metric_pattern, metric_direction
    "task": dict,          # must have description_file
    "stages": str,
}

TASK_YAML_RECOMMENDED = {
    "model_endpoint": dict,
    "container": dict,     # must have name, gpu, shm_size, devices
    "kimi_cli": dict,
    "max_retries_per_stage": int,
    "max_total_hours": (int, float),
}

BENCHMARK_REQUIRED = {
    "command": str,
    "metric_name": str,
    "metric_pattern": str,
    "metric_direction": str,
}

CONTAINER_REQUIRED = {
    "name": str,
    "gpu": str,
    "shm_size": str,
    "devices": list,
}

# metadata.json canonical schema
METADATA_REQUIRED = {
    "name": str,
    "category": str,       # bugfix | optimize | feature
    "difficulty": str,     # easy | medium | medium-hard | hard
    "source": dict,        # repo, pr/issue, merge_commit
    "description": str,
}

METADATA_RECOMMENDED = {
    "bug_type": str,       # regression | persistent | aiter-dependency
    "bug_source": str,     # sglang | vllm | aiter
    "commit_window": dict, # introducing_pr, fix_pr, pin_commit
    "aiter_dependency": bool,
    "model_required": bool,
    "gpu_required": (int, bool),
    "harness_type": str,   # unit_test | server_test | e2e_test
    "runtime_family": str, # sgl-dev | vllm-dev (determines Python path, workspace layout)
    "affected_files": list,
    "tags": list,
    "verified": dict,      # date, by, preflight_score, env
}

SOURCE_REQUIRED = {
    "repo": str,
}

# Valid values
VALID_TYPES = {"bugfix", "optimize", "feature"}
VALID_DIFFICULTIES = {"easy", "medium", "medium-hard", "hard"}
VALID_BUG_TYPES = {"regression", "persistent", "aiter-dependency"}
VALID_HARNESS_TYPES = {"unit_test", "server_test", "e2e_test"}
VALID_METRIC_DIRECTIONS = {"higher", "lower"}

# Python paths per image family
PYTHON_PATHS = {
    "sglang": "/opt/venv/bin/python3",
    "sgl-dev": "/opt/venv/bin/python3",
    "vllm": "/usr/bin/python3",
    "vllm-dev": "/usr/bin/python3",
}

# Unsafe patterns to flag
UNSAFE_PATTERNS = {
    "task_description.md": [
        (r"unset\s+PYTHONPATH", "Contains 'unset PYTHONPATH' (lesson #2)"),
        (r"kill\s+-9\s+\$\(pgrep", "Uses unsafe kill pattern 'kill -9 $(pgrep ...)' (lesson #4)"),
        (r"pkill\s+-9\s+python", "Uses broad 'pkill -9 python' (lesson #4)"),
    ],
    "test_harness.py": [
        (r"kill\s+-9\s+\$\(pgrep", "Uses unsafe kill pattern 'kill -9 $(pgrep ...)' (lesson #4)"),
        (r"pkill\s+-9\s+python", "Uses broad 'pkill -9 python' (lesson #4)"),
        (r'env\.pop\(["\']PYTHONPATH', "Removes PYTHONPATH from env (lesson #2)"),
        (r'env\[.PYTHONPATH.\]\s*=\s*""', "Blanks PYTHONPATH (lesson #2)"),
        (r"except\s+Exception\s*:", "Broad 'except Exception' — may mask target bug"),
    ],
    "Dockerfile": [
        (r"unset\s+PYTHONPATH", "Contains 'unset PYTHONPATH' in Dockerfile"),
    ],
}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

class ValidationResult:
    """Collects errors and warnings from validation."""

    def __init__(self, instance_name: str):
        self.instance_name = instance_name
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.instance_name}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        if self.passed and not self.warnings:
            lines.append("  All checks passed.")
        return "\n".join(lines)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file. Uses simple parser to avoid PyYAML dependency."""
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except ImportError:
        # Fallback: try to parse simple YAML-like structure
        # This is a best-effort parser for flat/shallow YAML
        result: dict[str, Any] = {}
        current_key = None
        current_dict: dict[str, Any] | None = None

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Top-level key with value
            if not line.startswith(" ") and not line.startswith("\t"):
                if ":" in stripped:
                    key, _, val = stripped.partition(":")
                    val = val.strip()
                    if val:
                        # Inline value
                        if val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        elif val.startswith("'") and val.endswith("'"):
                            val = val[1:-1]
                        elif val.startswith("["):
                            # List
                            try:
                                val = json.loads(val.replace("'", '"'))
                            except json.JSONDecodeError:
                                pass
                        elif val.isdigit():
                            val = int(val)
                        result[key.strip()] = val
                        current_key = None
                        current_dict = None
                    else:
                        # Nested dict starts
                        current_key = key.strip()
                        current_dict = {}
                        result[current_key] = current_dict
            elif current_dict is not None:
                if ":" in stripped:
                    key, _, val = stripped.partition(":")
                    val = val.strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    elif val.startswith("'") and val.endswith("'"):
                        val = val[1:-1]
                    elif val.startswith("["):
                        try:
                            val = json.loads(val.replace("'", '"'))
                        except json.JSONDecodeError:
                            pass
                    current_dict[key.strip()] = val

        return result


def validate_files(instance_dir: Path, result: ValidationResult) -> None:
    """Check that all required files exist."""
    for f in REQUIRED_FILES:
        if not (instance_dir / f).exists():
            result.error(f"Missing required file: {f}")

    for f in OPTIONAL_FILES:
        if not (instance_dir / f).exists():
            result.warn(f"Missing optional file: {f}")


def validate_task_yaml(instance_dir: Path, result: ValidationResult) -> dict[str, Any]:
    """Validate task.yaml structure and values."""
    path = instance_dir / "task.yaml"
    if not path.exists():
        return {}

    data = _load_yaml(path)

    # Required fields
    for field, expected_type in TASK_YAML_REQUIRED.items():
        val = data.get(field)
        if val is None:
            result.error(f"task.yaml: missing required field '{field}'")
        elif not isinstance(val, expected_type):
            result.warn(f"task.yaml: field '{field}' has unexpected type {type(val).__name__}, expected {expected_type.__name__}")

    # Recommended fields
    for field, expected_type in TASK_YAML_RECOMMENDED.items():
        val = data.get(field)
        if val is None:
            result.warn(f"task.yaml: missing recommended field '{field}'")

    # Validate type value
    task_type = data.get("type", "")
    if task_type and task_type not in VALID_TYPES:
        result.warn(f"task.yaml: type '{task_type}' not in {VALID_TYPES}")

    # Validate benchmark section
    benchmark = data.get("benchmark")
    if isinstance(benchmark, dict):
        for field, expected_type in BENCHMARK_REQUIRED.items():
            val = benchmark.get(field)
            if val is None:
                result.error(f"task.yaml: benchmark.{field} missing")

        # Check metric_pattern
        pattern = benchmark.get("metric_pattern", "")
        if pattern and "SCORE" not in pattern:
            result.warn(f"task.yaml: metric_pattern doesn't contain 'SCORE' — may not match harness output")

        # Check metric_direction
        direction = benchmark.get("metric_direction", "")
        if direction and direction not in VALID_METRIC_DIRECTIONS:
            result.error(f"task.yaml: metric_direction '{direction}' not in {VALID_METRIC_DIRECTIONS}")

        # Check command uses correct Python path based on base image
        # The python path depends on the Docker base image, not the repo:
        #   sgl-dev base → /opt/venv/bin/python3 (even for vLLM repos on sgl-dev)
        #   vllm-dev base → /usr/bin/python3
        command = benchmark.get("command", "")
        if command:
            base_image = data.get("base_image", "")
            # Also check Dockerfile FROM line as fallback
            dockerfile_base = ""
            dockerfile_path = instance_dir / "Dockerfile"
            if dockerfile_path.exists():
                for line in dockerfile_path.read_text().splitlines():
                    if line.strip().upper().startswith("FROM "):
                        dockerfile_base = line.strip().split()[1] if len(line.strip().split()) > 1 else ""
                        break
            # Determine expected python based on image family
            is_vllm_dev = "vllm-dev" in base_image or "vllm-dev" in dockerfile_base
            is_sgl_dev = "sgl-dev" in base_image or "sgl-dev" in dockerfile_base
            if is_vllm_dev and "/opt/venv/bin/python3" in command:
                result.error(f"task.yaml: benchmark.command uses sgl-dev python path but base image is vllm-dev")
            elif is_sgl_dev and "/usr/bin/python3" in command:
                result.error(f"task.yaml: benchmark.command uses vllm-dev python path but base image is sgl-dev")

        # Check preflight_timeout
        if benchmark.get("preflight_timeout") is None:
            result.warn("task.yaml: benchmark.preflight_timeout not set (recommend 120-1800)")

    # Validate container section
    container = data.get("container")
    if isinstance(container, dict):
        for field, expected_type in CONTAINER_REQUIRED.items():
            val = container.get(field)
            if val is None:
                result.warn(f"task.yaml: container.{field} missing")

    # Validate task.description_file
    task_section = data.get("task")
    if isinstance(task_section, dict):
        desc_file = task_section.get("description_file", "")
        if desc_file and not desc_file.startswith("/workspace/"):
            result.warn(f"task.yaml: task.description_file should be '/workspace/task_description.md', got '{desc_file}'")

    # Validate name matches directory
    name = data.get("name", "")
    dir_name = instance_dir.name
    if name and name != dir_name:
        result.warn(f"task.yaml: name '{name}' doesn't match directory name '{dir_name}'")

    return data


def validate_metadata(instance_dir: Path, result: ValidationResult) -> dict[str, Any]:
    """Validate metadata.json structure and values."""
    path = instance_dir / "metadata.json"
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        result.error(f"metadata.json: invalid JSON — {e}")
        return {}

    # Required fields
    for field, expected_type in METADATA_REQUIRED.items():
        val = data.get(field)
        if val is None:
            result.error(f"metadata.json: missing required field '{field}'")
        elif not isinstance(val, expected_type):
            result.warn(f"metadata.json: field '{field}' has unexpected type {type(val).__name__}")

    # Recommended fields
    for field, expected_type in METADATA_RECOMMENDED.items():
        val = data.get(field)
        if val is None:
            result.warn(f"metadata.json: missing recommended field '{field}'")

    # Validate values
    category = data.get("category", "")
    if category and category not in VALID_TYPES:
        result.warn(f"metadata.json: category '{category}' not in {VALID_TYPES}")

    difficulty = data.get("difficulty", "")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        result.warn(f"metadata.json: difficulty '{difficulty}' not in {VALID_DIFFICULTIES}")

    bug_type = data.get("bug_type", "")
    if bug_type and bug_type not in VALID_BUG_TYPES:
        result.warn(f"metadata.json: bug_type '{bug_type}' not in {VALID_BUG_TYPES}")

    harness_type = data.get("harness_type", "")
    if harness_type and harness_type not in VALID_HARNESS_TYPES:
        result.warn(f"metadata.json: harness_type '{harness_type}' not in {VALID_HARNESS_TYPES}")

    # Validate source section
    source = data.get("source")
    if isinstance(source, dict):
        if not source.get("repo"):
            result.error("metadata.json: source.repo missing")
        if not source.get("pr") and not source.get("issue"):
            result.warn("metadata.json: source has neither 'pr' nor 'issue'")
    elif source is not None:
        result.error(f"metadata.json: source should be dict, got {type(source).__name__}")

    # Check name consistency
    name = data.get("name", "")
    dir_name = instance_dir.name
    if name and name != dir_name:
        result.warn(f"metadata.json: name '{name}' doesn't match directory name '{dir_name}'")

    return data


def validate_content_safety(instance_dir: Path, result: ValidationResult) -> None:
    """Check file contents for unsafe patterns (lessons #2, #4)."""
    for filename, patterns in UNSAFE_PATTERNS.items():
        path = instance_dir / filename
        if not path.exists():
            continue
        content = path.read_text()
        for pattern, message in patterns:
            if re.search(pattern, content):
                if "except Exception" in message:
                    # This is a warning, not an error — it needs human review
                    result.warn(f"{filename}: {message}")
                else:
                    result.error(f"{filename}: {message}")


def validate_harness_output(instance_dir: Path, result: ValidationResult) -> None:
    """Check that test_harness.py prints SCORE in the expected format."""
    path = instance_dir / "test_harness.py"
    if not path.exists():
        return
    content = path.read_text()
    if "SCORE:" not in content:
        result.error("test_harness.py: does not contain 'SCORE:' output")
    if "SCORE: 0.0" not in content and "SCORE: {" not in content and 'f"SCORE:' not in content:
        result.warn("test_harness.py: may not have early-exit SCORE: 0.0 for critical failures")


def validate_dockerfile(instance_dir: Path, result: ValidationResult,
                        task_data: dict[str, Any]) -> None:
    """Validate Dockerfile structure."""
    path = instance_dir / "Dockerfile"
    if not path.exists():
        return
    content = path.read_text()

    # Check base image
    if "FROM" not in content:
        result.error("Dockerfile: no FROM instruction")

    # Check that test_harness.py and task_description.md are COPYed
    if "COPY test_harness.py" not in content:
        result.warn("Dockerfile: does not COPY test_harness.py")
    if "COPY task_description.md" not in content:
        result.warn("Dockerfile: does not COPY task_description.md")

    # Check for safe kill script (recommended)
    if "safe-kill-server" not in content:
        result.warn("Dockerfile: no safe-kill-server script installed (lesson #4)")

    # vLLM-specific checks
    repo = task_data.get("repo", "")
    if "vllm" in (repo or "").lower():
        if "pip uninstall" not in content and "PYTHONPATH" not in content:
            result.warn("Dockerfile: vLLM image but no wheel removal or PYTHONPATH (lessons #1, #3)")


def validate_instance(instance_dir: Path) -> ValidationResult:
    """Run all validation checks on an instance directory."""
    result = ValidationResult(instance_dir.name)

    validate_files(instance_dir, result)
    task_data = validate_task_yaml(instance_dir, result)
    validate_metadata(instance_dir, result)
    validate_content_safety(instance_dir, result)
    validate_harness_output(instance_dir, result)
    validate_dockerfile(instance_dir, result, task_data)

    return result


# ---------------------------------------------------------------------------
# Schema dump
# ---------------------------------------------------------------------------

def dump_schema() -> dict[str, Any]:
    """Return the canonical schema as a JSON-serializable dict."""
    return {
        "required_files": REQUIRED_FILES,
        "optional_files": OPTIONAL_FILES,
        "task_yaml": {
            "required": {k: v.__name__ if isinstance(v, type) else str(v) for k, v in TASK_YAML_REQUIRED.items()},
            "recommended": {k: v.__name__ if isinstance(v, type) else str(v) for k, v in TASK_YAML_RECOMMENDED.items()},
            "benchmark_required": {k: v.__name__ for k, v in BENCHMARK_REQUIRED.items()},
            "container_required": {k: v.__name__ for k, v in CONTAINER_REQUIRED.items()},
        },
        "metadata_json": {
            "required": {k: v.__name__ if isinstance(v, type) else str(v) for k, v in METADATA_REQUIRED.items()},
            "recommended": {k: v.__name__ if isinstance(v, type) else str(v) for k, v in METADATA_RECOMMENDED.items()},
            "source_required": {k: v.__name__ for k, v in SOURCE_REQUIRED.items()},
            "valid_values": {
                "category": sorted(VALID_TYPES),
                "difficulty": sorted(VALID_DIFFICULTIES),
                "bug_type": sorted(VALID_BUG_TYPES),
                "harness_type": sorted(VALID_HARNESS_TYPES),
                "metric_direction": sorted(VALID_METRIC_DIRECTIONS),
            },
        },
        "content_safety": {
            filename: [{"pattern": p, "message": m} for p, m in patterns]
            for filename, patterns in UNSAFE_PATTERNS.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate eval instance(s) against canonical schema"
    )
    parser.add_argument("instance_dirs", nargs="*",
                        help="Instance directory path(s) to validate")
    parser.add_argument("--all", action="store_true",
                        help="Validate all instances in evals/instances/")
    parser.add_argument("--dump-schema", action="store_true",
                        help="Dump canonical schema as JSON and exit")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    if args.dump_schema:
        print(json.dumps(dump_schema(), indent=2))
        return

    # Collect instance directories
    dirs: list[Path] = []
    if args.all:
        evals_dir = Path(__file__).resolve().parent.parent / "instances"
        dirs = sorted([d for d in evals_dir.iterdir() if d.is_dir()])
    elif args.instance_dirs:
        dirs = [Path(d) for d in args.instance_dirs]
    else:
        parser.error("Specify instance directory(ies) or --all")

    # Validate each
    all_passed = True
    for d in dirs:
        if not d.is_dir():
            print(f"[SKIP] {d} — not a directory")
            continue
        result = validate_instance(d)
        print(result.summary())
        print()
        if not result.passed:
            all_passed = False
        elif args.strict and result.warnings:
            all_passed = False

    # Summary
    total = len(dirs)
    passed = sum(1 for d in dirs if d.is_dir() and validate_instance(d).passed)
    print(f"{'=' * 60}")
    print(f"Total: {total}  Passed: {passed}  Failed: {total - passed}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
