# Optimize Qwen3-30B-A3B Pretraining MFU on MI355X

## Objective

Maximize TFLOP/s/GPU for Qwen3-30B-A3B (MoE) pretraining on 8x AMD Instinct MI355X GPUs using Primus. Current baseline is ~300 TFLOP/s/GPU (~740ms/iter). Push beyond config tuning into source-code-level optimizations.

## Model Architecture

- **Model**: Qwen3-30B-A3B
- **Type**: Mixture of Experts (MoE)
- **Experts**: 128 total, top-8 routing per token
- **Layers**: 48 (all MoE layers)
- **Hidden size**: 2048
- **FFN hidden**: 6144 (dense), 768 (per expert)
- **Attention**: GQA with 32 heads, 4 KV groups, 128 head dim
- **Vocab**: 151936

## Hardware & Configuration

- **GPUs**: 8x MI355X (288 GB HBM each)
- **Parallelism**: TP=1, EP=8, PP=1
- **Batch**: micro_batch_size=1, global_batch_size=8, seq_length=8192
- **Precision**: BF16
- **No gradient accumulation**
- **Memory usage**: ~145 GB / 288 GB (50%)

## Framework

- Primus + Primus-Turbo + Megatron-LM
- Source at `/workspace/primus_train/`
- Primus-Turbo provides CK/hipBLAS grouped GEMM, optimized attention, DeepEP

## Enabled Optimizations (Baseline)

All major optimizations are already ON:
- Turbo Attention (flash attention)
- Turbo Grouped MLP (auto-tuned GEMM backend)
- Turbo DeepEP (optimized MoE dispatch/combine)
- Sync-Free MoE Stage 1 (fused router + permute)
- Fused RoPE
- Fused Cross-Entropy (TE implementation)
- Precision-Aware Optimizer (bf16 states)
- Activation Recompute (5 layers, full/block)

## Prior Observations (Known Regressions)

- **Sync-Free MoE Stage 2**: REGRESSED throughput — do not enable
- **turbo_parallel_linear**: REGRESSED throughput — do not enable
- **TP=2 x EP=4**: ~9x slower due to TP all-reduce overhead — do not use

## Benchmark Command

```bash
cd /workspace/primus_train/Primus
./primus-cli direct \
  -- train pretrain --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml \
  --train_iters 10 \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --seq_length 8192 \
  --max_position_embeddings 8192 \
  --expert_model_parallel_size 8 \
  --mock_data True \
  --disable_last_saving True \
  --moe_use_legacy_grouped_gemm True \
  --use_turbo_grouped_mlp True \
  --use_turbo_attention True \
  --enable_primus_turbo True \
  --use_turbo_deepep True \
  --turbo_deepep_num_cu 80 \
  --turbo_sync_free_moe_stage 1 \
  --enable_experimental True \
  --apply_rope_fusion True \
  --cross_entropy_fusion_impl te \
  --cross_entropy_loss_fusion True \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  --recompute_num_layers 5 \
  --recompute_granularity full \
  --recompute_method block \
  --disable_wandb True \
  --disable_tensorboard True
```

## Profiling

Add `--profile` flag to the baseline script:
```bash
bash scripts/run_qwen3_30b_mfu_baseline.sh --profile
```
Trace saved under `output/amd/root/qwen3_30B_A3B-pretrain/tensorboard/`.

## Optimization Targets

Focus on SOURCE-CODE-LEVEL changes (not config tuning):
1. Profile the baseline to identify top bottlenecks
2. Analyze why mbs=1/seq=8192 underutilizes compute vs mbs=4/seq=4096
3. Optimize MoE GEMM kernel selection for small-batch expert shapes
4. Investigate communication/compute overlap opportunities
5. Router/top-k/permutation path optimization
