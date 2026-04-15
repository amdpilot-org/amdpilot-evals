# sglang-speculative-decode-fix

## Problem

Qwen3.5-397B-A17B-FP8 generates garbage or nonsensical output when served with
EAGLE speculative decoding enabled on ROCm GPUs. The output contains repeated
patterns of uppercase characters, broken tokens, and other artifacts that
indicate incorrect computation during the decode phase.

## Reproduction

Start an SGLang server with speculative decoding:

```bash
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b \
    --tp 2 \
    --attention-backend aiter \
    --mem-fraction-static 0.80 \
    --port 30000
```

Send any chat completion request. The response will be garbled or the server
may crash.

## Environment

- SGLang v0.5.9 (ROCm 7.2, MI300X)
- Model: Qwen/Qwen3.5-397B-A17B-FP8
- aiter GPU compute library (pre-installed in `/sgl-workspace/aiter`)

## Expected Behavior

The model produces coherent, correct responses to standard prompts. For
example, asking "What is 2 + 2?" should return a response containing "4".

## Debugging Clues

The crash manifests as `HSA_STATUS_ERROR_EXCEPTION: code 0x1016` (hardware
exception) in the scheduler process. The Python stack trace shows:

```
scheduler.py event_loop_overlap
  → eagle_worker_v2.py forward_batch_generation
    → sampler.py top_k_top_p_min_p_sampling_from_probs_torch (torch.multinomial)
```

The `torch.multinomial` call is where the async GPU exception is **detected**,
but the actual crash originates from an earlier GPU kernel launched during the
model forward pass (the last operations logged before the crash are aiter
fused_moe kernel configurations).

**Key isolation test**: EAGLE speculative decoding works correctly **without**
`SGLANG_ENABLE_SPEC_V2=1`. The V2 flag enables `eagle_worker_v2.py` and
overlap scheduling (`event_loop_overlap`), which runs GPU batches concurrently.
Try `--disable-overlap-schedule` to confirm. Check `server_args.py` around line
2955 for how SPEC_V2 enables overlap scheduling.

## Evaluation Criteria

The test harness starts the server and sends multiple prompts. All responses
must be coherent (no garbled text, no crashes). The score reflects the fraction
of prompts that receive valid, non-garbled responses.
