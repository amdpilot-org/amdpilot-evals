# Optimize Kimi K2.5 fused_moe_triton Performance

## Context

Kimi K2.5 is a multimodal encoder-decoder model (architectures: KimiK25ForConditionalGeneration, based on DeepseekV3) that uses Mixture of Experts (MoE).

When serving Kimi K2.5 with sglang, the fused_moe_triton kernel uses default config, resulting in poor performance. The model uses int4_w4a16 quantization with 384 experts.

Profiling shows the fused_moe kernel is a significant bottleneck:
- Prefill first MoE: ~9.11ms, second MoE: ~4.28ms
- Decode first MoE: ~501us, second MoE: ~180us

The fused_moe_triton kernel has a config lookup mechanism that loads tuned configurations based on model parameters (E, N, dtype), but the appropriate tuned configs may be missing for this model's specific configuration.

## Task

Investigate and resolve the poor fused_moe_triton performance for Kimi K2.5. The solution should ensure the kernel uses optimized configurations rather than defaults.

## Key Model Parameters

- `num_local_experts: 384` (E=384)
- `moe_intermediate_size: 2048`
- Quantization: int4_w4a16 (4-bit weights, group_size=32)
- Served with TP=8

## Required Approach

**You MUST follow this workflow — do not skip steps:**

1. **Read the MoE config tuning reference**: Read `/workspace/skills/amd-kernel-optimization/references/moe-config-tuning.md` — it explains how config lookup works, what parameters mean, and the correct tuning workflow.

2. **Understand the config lookup mechanism**: Read `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` to understand how configs are loaded. Check what config files exist and what's missing.

3. **Read the existing tuning infrastructure**: The `benchmark/kernels/fused_moe_triton/` directory contains:
   - `tuning_fused_moe_triton.py` — main tuning script with ray-based config search
   - `tuning_fused_moe_triton_sep.py` — separate up/down projection tuning
   - `common_utils.py` — shared utilities for config generation
   - `README.md` — usage instructions

   **Read these files BEFORE writing any configs.** They contain the correct methodology.

4. **Run systematic config benchmarking**: Either use the existing tuning script or write a systematic config search that tests multiple configs per batch size. Do NOT fabricate config values — benchmark each config to find the actual best.

5. **Benchmark with exclusive GPU access**: When running final benchmarks, ensure no other GPU-intensive processes are running.

## Environment

- Repository: sgl-project/sglang (code at `/workspace/sglang`)
- Docker container with ROCm, PyTorch, AMD GPU (8x MI300X)
- Use `/opt/venv/bin/python3` for all commands
- Model weights at `/sgl-workspace/models/models--moonshotai--Kimi-K2.5/`
- The benchmark/kernels/fused_moe_triton/ directory contains tuning scripts and utilities

## Verification

Run the test harness after applying your fix:
```bash
cd /workspace/sglang && /opt/venv/bin/python3 /workspace/test_harness.py
```
