# MoE Models Crash on ROCm with NameError

All MoE models (Kimi-K2.5, Mixtral-8x7B-FP8, gpt-oss) crash on AMD ROCm GPUs with:

```
NameError: name 'get_global_server_args' is not defined
```

The error occurs in `fused_experts_impl()` inside `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`. The error was introduced by commit `ee5ccde`.

## Affected File

- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Rules

- Edit files only under `/workspace/sglang`
- Use `/opt/venv/bin/python3` for all commands
- Do not modify `/workspace/test_harness.py`
- If you start auxiliary processes, never use broad kill patterns such as
  `pkill -f python` or `pkill -f sglang`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
