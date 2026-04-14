# FP8 GEMM Dispatch Missing DeepSeek-V3.2 Shape on gfx950

## Symptom

DeepSeek-V3.2 accuracy drops to 0% when running on MI355 (gfx950) GPUs.

## Root Cause Area

The FP8 block-scaled GEMM dispatch logic in the FP8 quantization dispatch module does not select the Triton kernel for the shape (N=7168, K=2304). Specifically, `use_aiter_triton_gemm_w8a8_tuned_gfx950(n=7168, k=2304)` returns `False`, causing the runtime to fall back to the CK (Composable Kernel) path.

The CK kernel produces incorrect output for this particular shape, which results in completely wrong model predictions and 0% accuracy on downstream tasks.

## Expected Behavior

The dispatch logic should route the (N=7168, K=2304) shape to the Triton kernel on gfx950, producing correct FP8 GEMM results and normal model accuracy.
