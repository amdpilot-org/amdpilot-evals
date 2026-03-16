# KernelBench x amdpilot — Triton on AMD MI355X

Evaluate amdpilot's multi-agent system on the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) benchmark (250 GPU kernel optimization problems) using Triton backend on AMD Instinct MI355X GPUs.

## Architecture

```
Executor (Qwen3.5-397B-A17B via SGLang)
  └─ Writes Triton kernels inside Docker container

Supervisor (Claude Opus 4.6 via proxy)
  └─ Reviews trials, adapts strategy, decides retry/stop

Nudge (Claude Opus 4.6 via proxy)
  └─ Monitors executor in real-time, provides AMD-specific guidance
```

## Prerequisites

### 1. Model Servers

| Role | Model | Endpoint |
|------|-------|----------|
| Executor | Qwen3.5-397B-A17B | `http://10.235.27.218:30000/v1` (SGLang) |
| Supervisor + Nudge | Claude Opus 4.6 | `http://localhost:8083/v1` (supervisor proxy) |

**Verify before running:**
```bash
# Executor
curl -s http://10.235.27.218:30000/v1/models

# Supervisor proxy (must be running in tmux session 'supervisor_proxy')
curl -s http://localhost:8083/health
```

### 2. Docker Image

The base Docker image `amdpilot-kernelbench-base:latest` must exist. It's built from `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315` with KernelBench + dependencies pre-installed.

```bash
# If the image doesn't exist, rebuild:
docker run -d --name kb-setup \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --shm-size 64g --network host \
    -v /home/jinpan12/KernelBench:/workspace/KernelBench \
    rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315 sleep infinity

docker exec kb-setup bash -c '
    cd /workspace/KernelBench
    /opt/venv/bin/pip install --no-deps -e .
    /opt/venv/bin/pip install pydra-config litellm openai datasets tqdm modal
'
# Copy the test harness
docker cp evals/kernelbench/test_harness.py kb-setup:/workspace/test_harness.py
docker commit kb-setup amdpilot-kernelbench-base:latest
docker rm -f kb-setup
```

### 3. KernelBench Repo

```bash
# Clone if not present
cd /home/jinpan12
git clone https://github.com/ScalingIntelligence/KernelBench.git
```

**Required patches** (already applied in our clone):
- `src/kernelbench/utils.py`: Changed `SERVER_PRESETS["local"]` to use chat completions API and configurable server address via `KERNELBENCH_SERVER_ADDRESS` env var
- `src/kernelbench/utils.py`: Fixed `query_server()` for local server type to always use `client.chat.completions.create()` (Qwen3.5 is a chat model, text completions returns empty)
- `scripts/generate_samples.py`: Fixed `problem_number` -> `work.problem_id` bug
- `scripts/eval_from_generations.py`: Added `ref_runtime` and `ref_runtime_stats` to eval results JSON

## Running

### Phase 2: Lightweight First Pass (single-shot, no amdpilot pipeline)

Generates Triton kernels for all 250 problems using a single LLM call per problem, then evaluates on MI355X.

```bash
bash evals/kernelbench/run_all.sh [run_name]
```

### Phase 3: Full amdpilot Pipeline (Supervisor + Executor + Nudge)

Re-runs failed problems from Phase 2 through the full multi-agent pipeline.

```bash
# 1. Generate task instances for failed problems
python3 evals/kernelbench/generate_tasks.py

# 2. Launch in tmux (persists after disconnect)
export NUM_WORKERS=4 MAX_HOURS_PER_PROBLEM=1
bash evals/kernelbench/run_phase3_tmux.sh
```

**IMPORTANT — `--frontier-model` flag:**
The worker script uses `--frontier-model` to route Supervisor and Nudge to Claude Opus 4.6. This requires:
- `AMDPILOT_SUPERVISOR_MODEL_URL=http://localhost:8083/v1` (set by the script)
- `AMDPILOT_SUPERVISOR_MODEL=claude-opus-4-6` (set by the script)
- The supervisor proxy must be running on port 8083 (check: `curl localhost:8083/health`)

Without `--frontier-model`, Supervisor and Nudge will fall back to the executor model (Qwen3.5), which is significantly less effective for reviewing and guiding.

### Monitoring

```bash
# Attach to tmux session
tmux attach -t kernelbench-phase3

# Count completed tasks
ls ~/amdpilot/evals/kernelbench/phase3_results/*.done | wc -l

# Live worker logs
tail -f ~/amdpilot/evals/kernelbench/phase3_results/worker_*.log

# Quick progress summary
python3 evals/kernelbench/compile_results.py
```

## Key Files

| File | Purpose |
|------|---------|
| `run_all.sh` | Phase 2: single-shot generation + eval for all 250 problems |
| `run_phase3_tmux.sh` | Phase 3: full pipeline in tmux with 4 workers |
| `generate_tasks.py` | Generates task.yaml + task_description.md per failed problem |
| `test_harness.py` | Generic KernelBench eval harness for amdpilot (SCORE: 0-100) |
| `compile_results.py` | Compiles per-level fast_p metrics from eval results |
| `analyze_results.py` | Detailed analysis with error distribution |
| `instances/` | Auto-generated task instances (one dir per problem) |
| `phase3_results/` | Full pipeline results (one dir per problem + worker logs) |

## Common Issues

### Nudge/Supervisor timeouts
If all 4 workers' Nudge agents show `timed out`, the model server is overloaded. The executor continues working — nudge timeouts don't cause task failure.

### Model server down → "no score" results
If the model server goes down mid-run, tasks get 0 trials / no metric. Fix: restart the server, then clear `.done` markers for affected tasks and re-run.

```bash
# Clear no-score markers to retry
python3 -c "
import json, os
d = 'evals/kernelbench/phase3_results'
for f in os.listdir(d):
    if not f.endswith('.done'): continue
    name = f.replace('.done','')
    s = os.path.join(d, name, 'summary.json')
    if os.path.isfile(s):
        data = json.load(open(s))
        if data.get('best_metric') in ('N/A', None):
            os.remove(os.path.join(d, f))
            print(f'Cleared {f}')
"
```

### `tl.math.tanh` / `tl.libdevice.*` not available
ROCm Triton does not support `tl.math.tanh` or `tl.libdevice.*`. The task descriptions include the manual workaround. If the executor still uses them, the Nudge agent (Claude) should catch and correct this.

### Eval results overwritten across levels
KernelBench uses a single `eval_results.json` per run directory. Problem IDs overlap across levels (1-100 for both L1 and L2). **Always rename `eval_results.json` to `eval_results_level{N}.json` after each level's evaluation**, or they will collide.
