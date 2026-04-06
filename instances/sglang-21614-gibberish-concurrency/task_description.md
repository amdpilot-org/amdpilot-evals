# Fix Gibberish Output at High Batch Concurrency (Issue #21614)

## Problem

When running `Qwen/Qwen3.5-397B-A17B-FP8` on 4× AMD Instinct MI355X GPUs with TP=4, the model produces gibberish output (responses starting with `!` followed by random characters) when batch concurrency reaches >= 32 concurrent requests. At concurrency levels 1–16, outputs are correct.

Adding `--disable-overlap-schedule` to the server launch command works around the issue, suggesting the bug is in the overlap scheduling logic.

## Environment

- **Runtime SGLang source**: `/sgl-workspace/sglang/` (editable install — edit files here)
- **Workspace symlink**: `/workspace/sglang` → `/sgl-workspace/sglang`
- **Python**: `/opt/venv/bin/python3`
- **GPU**: 4× AMD Instinct MI355X
- **ROCm**: 7.2
- **Model**: `Qwen/Qwen3.5-397B-A17B-FP8` (available via HuggingFace cache mount at `/root/.cache/huggingface`)

## Reproduction

Start the server:
```bash
/opt/venv/bin/python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
    --tp 4 \
    --trust-remote-code \
    --attention-backend triton \
    --disable-radix-cache \
    --mem-fraction-static 0.80 \
    --max-mamba-cache-size 128 \
    --host 0.0.0.0 --port 30000
```

Then run the test harness which sends requests at increasing concurrency levels (1, 4, 16, 32, 64) and checks whether any responses contain gibberish (starting with `!`).

**Bug behavior**: Concurrency 1–16 pass, concurrency 32+ produces gibberish.
**Expected behavior**: All concurrency levels produce valid, coherent output.

## Key Files to Investigate

The bug is in the overlap scheduling logic. Start looking at:
- `python/sglang/srt/managers/schedule_batch.py` — batch scheduling
- `python/sglang/srt/managers/tp_worker_overlap_thread.py` — overlap thread worker
- `python/sglang/srt/layers/attention/` — attention backends

## Rules

- Edit files ONLY under `/sgl-workspace/sglang/`
- Do NOT run `pip install -e .` — this would overwrite ROCm PyTorch
- Use `/opt/venv/bin/python3` for all Python commands
- Do NOT modify test harness or benchmark scripts
- Kill leftover server processes with safe pattern:
  `pgrep -f "python3 -m sglang" | xargs -r kill -9`
- Make clean, minimal fixes only — no debug prints

## Test Harness and Server Startup

The test harness (`/workspace/test_harness.py`) will:
1. Start the SGLang server with the model
2. Wait for it to be ready (this may take up to 15 minutes for a 397B model on 4 GPUs)
3. Run the concurrency sweep test
4. Report a SCORE from 0–100

**Important**: The model is very large (~397B params). Server startup may take 10–15 minutes. **Make the fix FIRST based on code analysis, then run the test harness ONCE to verify.** Do NOT repeatedly restart the server.

## Verification

After applying your fix, run:
```bash
pgrep -f "python3 -m sglang" | xargs -r kill -9; sleep 2
/opt/venv/bin/python3 /workspace/test_harness.py
```

A score of 100 means all concurrency levels produce clean (non-gibberish) output.
