# MTP Crash with FP4/FP8 Dispatch During CUDA Graph Capture

## Symptoms

When serving a DeepSeek model that uses MXFP4 quantized weights together with EAGLE
speculative decoding and MORI MoE dispatch enabled, the server crashes during CUDA
graph capture of the MTP (Multi-Token Prediction) layers.

The crash produces the following error:

```
Unsupported kernel config for moe heuristic dispatch
```

This error appears in the logs during the CUDA graph capture phase of server startup.
The server never finishes initializing and cannot serve any inference requests.

## Reproduction Conditions

The crash requires all of the following to be true simultaneously:

1. The main model uses MXFP4-quantized MoE expert weights.
2. EAGLE speculative decoding is enabled (which activates MTP layers).
3. MORI FP4/FP8 dispatch is active for MoE routing.
4. The MTP layers have BF16 weights (not quantized like the main model).

When these conditions are met, the MoE dispatch path encounters a combination of
activation and weight dtypes that it does not know how to handle, causing the crash.

## Goal

The fix should allow the MTP layers to function correctly even when the main model
uses a different quantization format. MTP layers use BF16 weights while the main
model uses MXFP4, and the MoE dispatch path must handle this difference.
