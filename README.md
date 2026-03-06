# amdpilot-evals

Evaluation instances for [amdpilot](https://github.com/Arist12/amdpilot). Each instance is a real-world AMD GPU task derived from actual GitHub PRs/issues.

## Scripts

### `scripts/curate_eval.py` -- PR-to-Eval Curation

Generates an eval instance from a merged GitHub PR:

```bash
python evals/scripts/curate_eval.py --pr https://github.com/sgl-project/sglang/pull/18903
python evals/scripts/curate_eval.py --pr sgl-project/sglang/18903 --generate-test
```

### `scripts/run_issue.py` -- Autonomous Issue Resolution

Resolves a GitHub issue end-to-end (env build + test generation + agent run):

```bash
python evals/scripts/run_issue.py https://github.com/sgl-project/sglang/issues/12345
# Or via CLI:
amdpilot issue https://github.com/sgl-project/sglang/issues/12345
```

### `scripts/enrich_registry.py` -- PR Registry Enrichment

Enriches lightweight PR rows into task-plane registry entries with replay and diff metadata:

```bash
python scripts/enrich_registry.py \
  --source /path/to/pr_rows.json \
  --output registry/enriched/sample.json \
  --diff-dir registry/ground_truth_diffs \
  --apply-check
```

Adds:

- canonical PR metadata from GitHub
- deterministic `replay_base_sha`
- normalized test commands
- optional ground-truth `.diff` export
- optional `git apply --check` verification

### `scripts/extract_validation_specs.py` -- Validation Spec Extraction

Builds task-plane validation specs from enriched PR rows:

```bash
python scripts/extract_validation_specs.py \
  --batch registry/enriched/sample.json \
  --output registry/enriched/sample.with_validation.json
```

Adds:

- validation tier (`1` / `2` / `3`)
- normalized runnable validation commands
- deterministic fallback checks for tier-2/3 tasks
- model-server bootstrap hints for serving workloads when recognized

## Data Curation Pipeline

### From a Merged PR

1. Fetch PR metadata via `gh` (title, body, files, merge commit)
2. Resolve parent commit (`merge_commit~1`) -- the repo state BEFORE the fix
3. Generate `task_description.md` from PR body (symptom only, no solution)
4. Generate `Dockerfile` using the ROCm base image + repo at parent commit
5. Generate `test_harness.py` (LLM-generated or manual)
6. Generate `task.yaml` with `stages: auto`

### From a PR Registry

For larger-scale task curation, the recommended flow is:

1. Prepare lightweight rows with at least `repo` and `pr_number`
2. Run `scripts/enrich_registry.py` to derive replay metadata and diffs
3. Run `scripts/extract_validation_specs.py` to attach validation contracts
4. Materialize selected rows into `instances/` as needed

### Data Leak Prevention

- Repo is checked out at `merge_commit~1` -- the fix does NOT exist
- `task_description.md` describes the symptom/bug only, never the solution code
- `test_harness.py` should test expected behavior, not specific code patterns

## Test Harness Quality

Test harnesses vary in reliability:

| Classification | Description | Instances |
|----------------|-------------|-----------|
| **STRONG** | Runtime checks (import, exec, GPU behavior, perf metrics) | fused-moe-fix, rotary-crash, flash-attn-overhead, qwen35-rope-fix |
| **MEDIUM** | Mix of static + runtime | mla-nhead8, vllm-fp4-hwdetect |
| **WEAK** | Static string/pattern matching only | eagle3-aiter-fix, aiter-pagesize-fix, vllm-encoder-rocm, vllm-ck-mxfp4-moe, deepseek-r1-optimize |

**Known limitation**: Weak test harnesses can be passed with cosmetic changes (comments, string additions) without actually fixing the bug. For production use, prefer instances with STRONG harnesses or add runtime validation.

## Base Image

All evals default to `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226`:
- `/opt/venv/bin/python3` with ROCm PyTorch, triton, aiter, composable kernel
- `/sgl-workspace/aiter` -- aiter source and compiled kernels
- Never use `pip install -e .` on target repos

## Available Instances (10)

| Instance | Category | Difficulty | Harness | Source |
|----------|----------|------------|---------|--------|
| sglang-fused-moe-fix | Bug Fix | Easy | STRONG | sglang #19840 |
| aiter-mla-nhead8 | Feature | Medium | MEDIUM | aiter #2138 |
| aiter-flash-attn-overhead | Performance | Medium-Hard | STRONG | aiter #2129 |
| sglang-rotary-crash | Bug Fix | Medium | STRONG | sglang #18903 |
| sglang-qwen35-rope-fix | Bug Fix | Medium | STRONG | sglang #18753 |
| sglang-deepseek-r1-optimize | Optimization | Medium-Hard | WEAK | sglang #18242 |
| vllm-encoder-rocm | Bug Fix | Hard | WEAK | vllm #35334 |
| vllm-ck-mxfp4-moe | Feature | Hard | WEAK | vllm #34301 |
| sglang-eagle3-aiter-fix | Bug Fix | Hard | WEAK | sglang #19362 |
| sglang-aiter-pagesize-fix | Bug Fix | Hard | WEAK | sglang #16531 |

## Running an Eval

```bash
# Build Docker image
cd evals/instances/<name> && docker build -t amdpilot-eval-<name> .

# Run (fully autonomous)
uv run amdpilot run evals/instances/<name>/task.yaml
```

## Testing

The repo's task-plane helper scripts have unit coverage:

```bash
python -m unittest discover -s tests -v
```
