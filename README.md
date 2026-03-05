# amdpilot-evals

Evaluation instances for [amdpilot](https://github.com/Arist12/amdpilot). Each instance is a real-world AMD GPU task derived from actual GitHub PRs/issues.

## Data Curation Pipeline

### Automated (recommended)

Use the curation script to generate an eval instance from a merged GitHub PR:

```bash
# Basic — generates task description, Dockerfile, YAML; test harness is a stub
python scripts/curate_eval.py --pr https://github.com/sgl-project/sglang/pull/18903

# With LLM-generated test harness (requires model endpoint)
python scripts/curate_eval.py --pr sgl-project/sglang/18903 --generate-test
```

The script:
1. Fetches PR metadata (title, body, files, merge commit) via `gh`
2. Determines the "before" state: `merge_commit~1` (the fix does NOT exist)
3. Generates `task_description.md` from the PR body, stripping solution code
4. Generates `Dockerfile` using the appropriate ROCm base image
5. Generates `task.yaml` with `stages: auto` (supervisor plans at runtime)
6. Optionally generates `test_harness.py` via LLM (or creates a stub for manual authoring)

### Manual

Create `evals/instances/<name>/` with these files:

| File | Purpose | Required |
|------|---------|----------|
| `metadata.json` | Category, difficulty, source PR | Yes |
| `task_description.md` | Bug symptom / feature spec (NO solution code) | Yes |
| `test_harness.py` | Verification script (SCORE: 0-100 output) | Yes |
| `Dockerfile` | Build eval Docker image | Yes |
| `task.yaml` | amdpilot job config | Yes |

## Data Leak Prevention

Every eval instance MUST follow these rules:

1. **Repo checked out at `merge_commit~1`** — the fix does not exist in the codebase
2. **task_description.md describes the SYMPTOM** — never include the solution code or diff
3. **test_harness.py tests BEHAVIOR** — check expected outcomes (no crash, correct output), not specific code patterns (no AST matching for the exact fix)

## Base Image Conventions

All evals default to `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226` which provides:
- `/opt/venv/bin/python3` with ROCm PyTorch, triton, aiter, composable kernel
- `/sgl-workspace/aiter` — aiter source and compiled kernels
- `/dev/kfd`, `/dev/dri` — AMD GPU access
- **Critical**: never `pip install -e .` on target repos

## Available Instances

| Instance | Category | Difficulty | Source |
|----------|----------|------------|--------|
| `sglang-fused-moe-fix` | Bug Fix | Easy | [sglang #19840](https://github.com/sgl-project/sglang/pull/19840) |
| `aiter-mla-nhead8` | Feature | Medium | [aiter #2138](https://github.com/ROCm/aiter/pull/2138) |
| `aiter-flash-attn-overhead` | Performance | Medium-Hard | [aiter #2129](https://github.com/ROCm/aiter/issues/2129) |

## Running an Eval

```bash
# 1. Build the Docker image
cd evals/instances/<name>
docker build -t amdpilot-eval-<name> .

# 2. Run with amdpilot (fully autonomous)
uv run amdpilot run evals/instances/<name>/task.yaml
```
