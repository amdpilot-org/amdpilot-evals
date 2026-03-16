#!/usr/bin/env python3
"""
Generate amdpilot task.yaml + task_description.md for KernelBench problems.

Creates eval instances under evals/instances/kernelbench-L{level}-P{pid}/
for each failed problem from the Phase 2 lightweight pass.
"""

import json
import os
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMDPILOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
KB_DIR = "/home/jinpan12/KernelBench"
RUN_DIR = os.path.join(KB_DIR, "runs", "amdpilot_triton_qwen35_v1")
INSTANCES_DIR = os.path.join(SCRIPT_DIR, "instances")
BASE_IMAGE = "amdpilot-kernelbench-base:latest"
MODEL_SERVER = os.environ.get("KERNELBENCH_SERVER_ADDRESS", "10.235.27.218")


def load_failed_problems():
    """Load all failed problems from Phase 2 eval results."""
    failed = []
    for level, fname in [
        (1, "eval_results_level1_v2.json"),
        (2, "eval_results_level2.json"),
        (3, "eval_results_level3.json"),
    ]:
        fpath = os.path.join(RUN_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        for pid, samples in data.items():
            s = samples[0]
            if not s["correctness"]:
                error_info = ""
                meta = s.get("metadata", {})
                for key in ["runtime_error", "cuda_error", "other_error", "error"]:
                    if key in meta:
                        error_info = str(meta[key])[:500]
                        break
                failed.append({
                    "level": level,
                    "problem_id": int(pid),
                    "compiled": s["compiled"],
                    "error": error_info,
                })
    return failed


def load_problem_code(level, problem_id):
    """Load the original PyTorch reference code for a problem."""
    level_dir = os.path.join(KB_DIR, "KernelBench", f"level{level}")
    if not os.path.exists(level_dir):
        return None
    for fname in os.listdir(level_dir):
        if fname.startswith(f"{problem_id}_"):
            with open(os.path.join(level_dir, fname)) as f:
                return f.read(), fname
    return None, None


def load_previous_attempt(level, problem_id):
    """Load the previous failed Triton kernel attempt if available."""
    kernel_path = os.path.join(
        RUN_DIR,
        f"level_{level}_problem_{problem_id}_sample_0_kernel.py",
    )
    if os.path.exists(kernel_path):
        with open(kernel_path) as f:
            return f.read()
    return None


TRITON_ROCM_GUIDANCE = """\
## AMD ROCm Triton Constraints (CRITICAL)

You are writing Triton kernels for AMD Instinct MI355X (gfx950, CDNA4) with ROCm.

### Known Issues - You MUST follow these rules:

1. **`tl.math.tanh` is UNAVAILABLE** on ROCm Triton. Use manual implementation:
   ```python
   x_clamped = tl.maximum(tl.minimum(x, 10.0), -10.0)
   exp_2x = tl.math.exp(2.0 * x_clamped)
   tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
   ```

2. **`tl.libdevice.*` is UNAVAILABLE** on ROCm. Do NOT use `tl.libdevice.tanh`,
   `tl.libdevice.exp`, etc. Use `tl.math.exp` or manual implementations.

3. **Wavefront size is 64** (not 32 like NVIDIA). BLOCK_SIZE values that are
   multiples of 64 align better with hardware.

4. **Cast output to target dtype explicitly**: compute in float32, cast back on store:
   ```python
   x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
   y = compute(x)
   tl.store(out_ptr, y.to(tl.float32), mask=mask)
   ```

5. **BLOCK_SIZE selection**: Use `triton.next_power_of_2(N)` for the hidden dimension.

6. **Common kernel pattern**:
   ```python
   @triton.jit
   def _kernel(X_ptr, Y_ptr, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
       row = tl.program_id(0)
       cols = tl.arange(0, BLOCK_SIZE)
       mask = cols < N
       x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
       y = compute(x)
       tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float32), mask=mask)
   ```
"""


def generate_task_description(level, problem_id, problem_code, problem_name,
                              prev_attempt, error_info):
    """Generate a task description markdown file."""
    desc = f"# KernelBench Level {level} Problem {problem_id}: {problem_name}\n\n"
    desc += "## Goal\n\n"
    desc += (
        "Write an optimized Triton kernel implementation (`ModelNew`) that:\n"
        "1. Produces the **exact same output** as the PyTorch reference `Model`\n"
        "2. Is **faster** than the PyTorch baseline\n"
        "3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)\n\n"
    )
    desc += "## PyTorch Reference Implementation\n\n```python\n"
    desc += problem_code
    desc += "\n```\n\n"
    desc += TRITON_ROCM_GUIDANCE + "\n"

    if prev_attempt:
        desc += "## Previous Failed Attempt\n\n"
        desc += "A previous single-shot attempt failed. "
        if error_info:
            desc += f"Error: `{error_info[:200]}`\n\n"
        else:
            desc += "The output was incorrect.\n\n"
        desc += "```python\n" + prev_attempt[:3000] + "\n```\n\n"
        desc += "Analyze what went wrong and fix the issues.\n\n"

    desc += "## Output Requirements\n\n"
    desc += (
        "Save your implementation to `/workspace/generated_kernel.py`.\n"
        "The file must define a `ModelNew` class with the same interface as `Model`.\n"
        "Run the test harness to verify:\n"
        "```bash\n"
        f"/opt/venv/bin/python3 /workspace/test_harness.py --level {level} --problem-id {problem_id}\n"
        "```\n"
    )
    return desc


def generate_task_yaml(level, problem_id, instance_dir, gpu_id="0"):
    """Generate a task.yaml for amdpilot."""
    task = {
        "name": f"kernelbench-L{level}-P{problem_id}",
        "type": "optimize",
        "base_image": BASE_IMAGE,
        "model_endpoint": {
            "base_url": f"http://{MODEL_SERVER}:30000/v1",
            "model": "Qwen3.5-397B-A17B",
            "api_key": "dummy",
        },
        "container": {
            "name": f"amdpilot_kb_L{level}_P{problem_id}",
            "gpu": gpu_id,
            "shm_size": "16g",
            "devices": ["/dev/kfd", "/dev/dri"],
            "volumes": [
                "/home/jinpan12/KernelBench:/workspace/KernelBench:ro",
            ],
        },
        "workload": {
            "description": f"Optimize KernelBench Level {level} Problem {problem_id} with Triton on AMD MI355X",
            "framework": "PyTorch",
        },
        "benchmark": {
            "command": f"/opt/venv/bin/python3 /workspace/test_harness.py --level {level} --problem-id {problem_id}",
            "metric_name": "score",
            "metric_pattern": r"SCORE:\s+([\d.]+)",
            "metric_direction": "higher",
        },
        "task": {
            "description_file": os.path.join(instance_dir, "task_description.md"),
        },
        "stages": "auto",
        "max_retries_per_stage": 2,
        "max_total_hours": 0.5,
    }

    import yaml
    return yaml.dump(task, default_flow_style=False, sort_keys=False)


def main():
    failed = load_failed_problems()
    print(f"Found {len(failed)} failed problems")

    os.makedirs(INSTANCES_DIR, exist_ok=True)

    generated = 0
    for prob in failed:
        level = prob["level"]
        pid = prob["problem_id"]
        instance_name = f"kernelbench-L{level}-P{pid}"
        instance_dir = os.path.join(INSTANCES_DIR, instance_name)
        os.makedirs(instance_dir, exist_ok=True)

        result = load_problem_code(level, pid)
        if result is None or result[0] is None:
            print(f"  Skipping L{level} P{pid}: problem code not found")
            continue
        code, pname = result
        prev_attempt = load_previous_attempt(level, pid)

        desc = generate_task_description(
            level, pid, code, pname, prev_attempt, prob["error"]
        )
        with open(os.path.join(instance_dir, "task_description.md"), "w") as f:
            f.write(desc)

        yaml_content = generate_task_yaml(level, pid, instance_dir)
        with open(os.path.join(instance_dir, "task.yaml"), "w") as f:
            f.write(yaml_content)

        generated += 1

    print(f"Generated {generated} task instances in {INSTANCES_DIR}")


if __name__ == "__main__":
    main()
