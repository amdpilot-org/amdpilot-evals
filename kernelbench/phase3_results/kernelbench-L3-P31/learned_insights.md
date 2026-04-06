# Learned Insights

- **Trial 1**: For KernelBench L3P31 (VisionAttention): torch.compile(mode='default') gives 1.5x speedup (11.7ms→7.31ms, score 65.10)
- **Trial 1**: torch.compile(mode='max-autotune') was slower (14.2ms) than mode='default' (7.31ms) on this workload due to runtime kernel search overhead
- **Trial 1**: nn.MultiheadAttention uses a fused in_proj_weight — to use F.scaled_dot_product_attention, split in_proj_weight into 3 chunks along dim=0 for Q/K/V
- **Trial 1**: Profiling breakdown: bmm ~40%, mm ~35%, elementwise ops ~25% — fusing layernorm+residual could save significant time
- **Trial 1**: torch.set_float32_matmul_precision('high') enables TF32-like tensor core usage on AMD MI355X
- **Trial 2**: Trial 2 produced no output — agent may have been stuck on planning without executing. Need very concrete step-by-step instructions.
- **Trial 2**: For nn.MultiheadAttention → F.scaled_dot_product_attention conversion: split in_proj_weight with chunk(3, dim=0), reshape to (B, num_heads, seq_len, head_dim) for SDPA
- **Trial 2**: inductor_config.coordinate_descent_tuning and max_autotune can improve torch.compile codegen quality
- **Trial 3**: Agent produced no output in trials 2 and 3 — needs extremely explicit, copy-paste-ready code and step-by-step execution commands
- **Trial 3**: For F.scaled_dot_product_attention: use F.linear(x, attn.in_proj_weight, attn.in_proj_bias) then chunk(3, dim=-1) to get Q/K/V, reshape to (B, num_heads, seq_len, head_dim) for SDPA
- **Trial 4**: Agent has failed to produce output in 3 consecutive trials (2,3,4) — needs complete copy-paste code blocks, not instructions
- **Trial 4**: For KernelBench tasks, providing the full generated_kernel.py as a heredoc in shell commands is the most reliable way to get the agent to execute
- **Trial 5**: Agent has failed to produce output in 4 consecutive trials (2-5) — must provide complete copy-paste shell commands with no ambiguity
- **Trial 5**: For F.scaled_dot_product_attention with nn.MultiheadAttention weights: use F.linear(x, attn.in_proj_weight, attn.in_proj_bias) then chunk(3, dim=-1), reshape to (B, num_heads, seq_len, head_dim)
