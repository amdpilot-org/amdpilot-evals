# Fix accuracy degradation running DeepSeek-R1-MXFP4 at TP=4 with MTP on AMD GPUs

## Issue

This is a known issue affecting AMD GPU deployments.

When running the DeepSeek-R1-MXFP4 model on AMD GPUs with tensor parallelism
degree 4 (TP=4) and Multi-Token Prediction (MTP) enabled, the model produces
subtly wrong outputs. There is no crash, no assertion failure, and no error
message -- the model appears to run normally but its outputs are degraded in
quality compared to TP=8 or non-MTP configurations.

## Symptoms

- **TP=8 works correctly**: Running the same model at TP=8 produces expected,
  high-quality outputs.
- **TP=4 produces wrong outputs**: At TP=4, outputs are subtly incorrect --
  not garbage, but noticeably degraded. The model may give wrong answers,
  produce incoherent reasoning chains, or otherwise exhibit accuracy loss.
- **Silent failure**: No errors, warnings, or crashes. The issue is purely in
  output quality, making it difficult to detect without careful validation.
- **MTP interaction**: The problem manifests when MTP (speculative decoding
  with multiple draft tokens) is enabled alongside TP=4.
- **AMD-specific**: The issue occurs on AMD GPUs using the aiter attention
  backend. It does not reproduce on NVIDIA hardware.

## Key Observations

- The issue only manifests at TP=4, not TP=8.
- The aiter attention backend's MLA decode path may not correctly handle
  all tensor parallelism configurations.
- This is a **cross-repo** issue: the fix may involve changes to both sglang
  source code and the aiter dependency version.

## Environment

- SGLang source is available at `/sgl-workspace/sglang`
- Workspace symlink: `/workspace/sglang`
- Use `/opt/venv/bin/python3`
- Do **not** blank or clear `PYTHONPATH`
- Kill any leftover server processes with:

```bash
pgrep -f "python3 -m sglang" | xargs -r kill -9 2>/dev/null; sleep 2
```

## Verification

Run exactly:

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server at TP=4 with the aiter attention backend,
sends inference requests, and checks that outputs are coherent and
factually correct. It reports a `SCORE` from `0` to `100`.

## Rules

- Edit source files only under `/sgl-workspace/sglang/`
- Do **not** modify `test_harness.py`
- Keep the patch minimal and localized
- Do **not** use broad `pkill` / `pgrep` patterns
- Note: this is a cross-repo bug. The aiter library version may also need
  to be updated (e.g., via pip install) in addition to sglang source changes.
