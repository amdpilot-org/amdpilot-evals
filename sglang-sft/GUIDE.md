# SFT Trajectory Collection: Claude Opus 4.6 Executor

How to use amdpilot with Claude Opus 4.6 as **all three agents** (supervisor, executor, nudge) to solve SGLang PRs and collect SFT trajectory data.

## Prerequisites

- **Supervisor proxy** running on port 8083. The proxy translates OpenAI-format requests (including tool/function calls) to Anthropic Messages format, enabling kimi-cli to use Claude Opus 4.6 without OAuth.
  ```bash
  # Check health:
  curl localhost:8083/health
  # If not running:
  cd /home/jinpan12/amdpilot
  source .env.supervisor
  python3 tools/supervisor_proxy.py &
  ```
- **GPUs** available (MI355X). Check with `rocm-smi --showid`
- **Models** on NFS at `/mnt/dcgpuval/` (mounted read-only into containers)
- **Branch**: `claude-executor-sft-data`

## Architecture

```
kimi-cli (executor inside Docker)
   ↓ OpenAI format (with tools)
supervisor_proxy.py :8083
   ↓ Anthropic Messages format (with tools)
https://llm-api.amd.com/Anthropic/v1/messages
   ↓
Claude Opus 4.6
```

All three agents — supervisor, executor, and nudge — use the same proxy.
The proxy handles two paths:
- **Without tools**: Routes to `claude3/{model}/chat/completions` (existing supervisor path)
- **With tools**: Translates OpenAI tool format → Anthropic tool format, routes to `Anthropic/v1/messages`

## Step-by-step: Add a New PR Experiment

### 1. Fetch the PR info

```bash
# Get the merge commit hash
gh pr view <PR_NUMBER> -R sgl-project/sglang --json mergeCommit,headRefOid

# Get the parent commit (the codebase state BEFORE the fix was merged)
docker exec amdpilot-kernelbench bash -c \
  "cd /sgl-workspace/sglang && git log --format='%H %s' <MERGE_COMMIT>~1 -1"
```

Write down:
- `PARENT_COMMIT`: the commit hash just before the fix (the buggy code)
- `PR_NUMBER`: the SGLang PR number

### 2. Create the issue directory

```bash
ISSUE=<PR_NUMBER>
mkdir -p evals/sglang-sft/issue-${ISSUE}/dist
cp -r dist/kimi-cli evals/sglang-sft/issue-${ISSUE}/dist/kimi-cli
```

### 3. Write `task_description.md`

Create `evals/sglang-sft/issue-${ISSUE}/task_description.md` with:

```markdown
# <PR Title>

## Bug Description
<Copy from the PR's "Motivation" section. Describe the bug or missing feature.>

## Expected Fix
<What the agent should do. Copy from the PR's "Modifications" section.>

## Environment
- **Hardware**: MI355 x8
- **Repository**: SGLang at `/sgl-workspace/sglang`
- **Key file(s)**: <path to the file(s) that need fixing>
- **Model**: <model name> at `/models/<model-name>`

## Reproduction
<Server launch command and steps to trigger the bug.>

## Verification
<How to verify the fix works. Include benchmark commands.>

The test harness at `/workspace/test_harness.py` will:
1. Start the server with the relevant flags
2. Send test requests
3. Report SCORE: 100 if working correctly
```

### 4. Write `test_harness.py`

Create `evals/sglang-sft/issue-${ISSUE}/test_harness.py` — a Python script that:
- Starts the SGLang server with the flags that reproduce the bug
- Waits for the server to come up (or detects a crash)
- Sends a simple test request
- Prints `SCORE: 100` if the server starts and responds, `SCORE: 0` if it crashes

See existing examples in `evals/sglang-sft/issue-19935/test_harness.py`.

### 5. Write the `Dockerfile`

Create `evals/sglang-sft/issue-${ISSUE}/Dockerfile`:

```dockerfile
FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315
SHELL ["/bin/bash", "-c"]

# Revert SGLang to the commit BEFORE the fix
RUN cd /sgl-workspace/sglang && \
    git stash && \
    git fetch origin && \
    git checkout <PARENT_COMMIT> && \
    /opt/venv/bin/pip install -e "python[all]" --no-deps 2>/dev/null || true

# Pre-install uv + kimi-cli (required — container has Python 3.10, kimi needs 3.12+)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY dist/kimi-cli/ /tmp/kimi-wheels/
RUN source $HOME/.local/bin/env && \
    uv run --python 3.14 \
      --with /tmp/kimi-wheels/kimi_cli-1.15.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/kosong-0.43.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/pykaos-0.7.0-py3-none-any.whl \
      -- kimi --version

# Copy test harness
COPY test_harness.py /workspace/test_harness.py
RUN chmod +x /workspace/test_harness.py

# Symlink model (adjust path for the model this PR needs)
RUN mkdir -p /models && \
    ln -sf /mnt/dcgpuval/<path-to-model> /models/<model-name>

WORKDIR /sgl-workspace/sglang
CMD ["sleep", "infinity"]
```

**Important**: The `COPY dist/kimi-cli/` line requires the wheels at `evals/sglang-sft/issue-${ISSUE}/dist/kimi-cli/`. That's why step 2 copies them.

### 6. Write `task.yaml`

Create `evals/sglang-sft/issue-${ISSUE}/task.yaml`:

```yaml
name: sglang-issue-<PR_NUMBER>
type: bugfix                    # or "feature" for feature PRs
repo: https://github.com/sgl-project/sglang.git
base_image: sglang-issue-<PR_NUMBER>:base

# All agents use Claude Opus 4.6 via the tool-translating proxy
model_endpoint:
  base_url: "http://localhost:8083/v1"
  model: "claude-opus-4-6"
  api_key: "sk-dummy"

supervisor_model_endpoint:
  base_url: "http://localhost:8083/v1"
  model: "claude-opus-4-6"
  api_key: "sk-dummy"

container:
  name: amdpilot_sglang_<PR_NUMBER>
  gpu: "0,1,2,3"               # Adjust for TP needs (4 GPUs for TP=4, 8 for TP=8)
  shm_size: 64g
  devices: [/dev/kfd, /dev/dri]
  volumes:
    - "/mnt/dcgpuval:/mnt/dcgpuval:ro"

workload:
  description: "<short description of the task>"
  framework: PyTorch

benchmark:
  command: "/opt/venv/bin/python3 /workspace/test_harness.py"
  metric_name: score
  metric_pattern: 'SCORE:\s+([\d.]+)'
  metric_direction: higher

task:
  description_file: evals/sglang-sft/issue-<PR_NUMBER>/task_description.md

stages: auto

kimi_cli:
  repo_url: "https://github.com/Arist12/kimi-cli.git"
  branch: amd-dev
  install_dir: "/sgl-workspace/kimi-cli"
  provider: sglang
  provider_type: openai_legacy
  thinking: true
  yolo: true
  ralph_iterations: -1
  max_steps_per_turn: 500
  reserved_context_size: 50000

max_retries_per_stage: 3
max_total_hours: 2
supervisor: true
nudge: true
```

### 7. Build the Docker image

```bash
cd /home/jinpan12/amdpilot
docker build --no-cache -t sglang-issue-<PR_NUMBER>:base evals/sglang-sft/issue-<PR_NUMBER>/
```

### 8. Launch the run

```bash
cd /home/jinpan12/amdpilot
export AMDPILOT_SUPERVISOR_MODEL_URL="http://localhost:8083/v1"
export AMDPILOT_SUPERVISOR_MODEL="claude-opus-4-6"

RESULTS_DIR="evals/sglang-sft/results/issue-<PR_NUMBER>-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

uv run amdpilot run evals/sglang-sft/issue-<PR_NUMBER>/task.yaml \
    --results-dir "$RESULTS_DIR" \
    --hours 2 \
    --gpu "0,1,2,3" \
    --frontier-model \
    2>&1 | tee "$RESULTS_DIR/run.log"
```

### 9. Collect trajectory data

After the run completes, trajectories are at:
```
$RESULTS_DIR/
  agent_output/
    trial_N.txt                        # Raw agent stdout
    trial_N_trajectory/
      prompt.txt                       # Prompt given to the executor
      sessions/<session-id>/<run-id>/
        context.jsonl                  # Multi-turn conversation (SFT data source)
        wire.jsonl                     # Raw wire messages
  summary.json                         # Score, verification, best metric
  scoreboard.jsonl                     # Per-trial metrics
  run.log                              # Full orchestrator log
```

The key file for SFT is `context.jsonl` — the full multi-turn trajectory with roles `user`, `assistant`, `tool`.

## Key Gotchas

1. **`provider_type` must be `openai_legacy`** in the task.yaml kimi_cli section. The proxy at port 8083 accepts OpenAI format and translates tools to Anthropic format internally. Do NOT use `provider_type: anthropic` — that triggers kimi-cli's OAuth login requirement.

2. **Do NOT set `AMDPILOT_EXECUTOR_USE_FRONTIER=1`** — This overrides the model_endpoint with the supervisor endpoint (a `SupervisorModelConfig` with `is_anthropic=True`), causing `_build_kimi_config_toml` to force `provider_type: anthropic` for kimi-cli. Instead, point `model_endpoint` directly at the proxy.

3. **kimi-cli must be pre-installed in the Docker image** — The base image has Python 3.10 but kimi-cli requires 3.12+. The orchestrator uses `uv run --python 3.14` but downloading Python + 157 packages at runtime can cause container crashes between retries. Pre-installing via `SHELL ["/bin/bash", "-c"]` + `RUN source ... && uv run ...` in the Dockerfile avoids this.

4. **Container uses `--network host`** — So `localhost:8083` inside the container reaches the host proxy.

5. **Model paths** — Mount NFS via the `volumes` section in task.yaml. Symlink to `/models/<name>` in the Dockerfile.

6. **The supervisor proxy must have tool translation support** — The updated `tools/supervisor_proxy.py` detects tool-bearing requests and routes them to the Anthropic Messages endpoint (`/Anthropic/v1/messages`) with proper format translation. Requests without tools go through the original `claude3/{model}/chat/completions` path.

## Available Models on This Machine

| Model | Path |
|---|---|
| Kimi-K2.5 | `/mnt/dcgpuval/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1` |
| Kimi-K2.5-MXFP4 | `/mnt/dcgpuval/huggingface/hub/models--amd--Kimi-K2.5-MXFP4/snapshots/<check>` |
| DeepSeek-R1-MXFP4 | `/mnt/dcgpuval/datasets/data/DeepSeek-R1-0528-MXFP4-Preview` |
| Kimi-K2-Thinking | `/mnt/dcgpuval/huggingface/hub/models--moonshotai--Kimi-K2-Thinking` |
| Kimi-K2-Instruct | `/data/moonshotai--Kimi-K2-Instruct` |

## File Structure

```
evals/sglang-sft/
  GUIDE.md                          # This file
  run_sft_collection.sh             # Batch launcher
  issue-19935/
    task_description.md
    task.yaml
    test_harness.py
    Dockerfile
    dist/kimi-cli/                  # Copied from amdpilot/dist/kimi-cli/
  issue-20187/
    ...same structure...
  results/
    issue-19935-<timestamp>/        # Per-run results with trajectories
    issue-20187-<timestamp>/
```
