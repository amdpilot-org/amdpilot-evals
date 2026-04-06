# amdpilot — kernelbench-L3-P37

| Property | Value |
|----------|-------|
| **Task** | Optimize KernelBench Level 3 Problem 37 with Triton on AMD MI355X |
| **Type** | optimize |
| **Duration** | 0.5h (1748s) |
| **Best metric** | 60.20 score |
| **Trials** | 1 (0 failed) |
| **Stages** | 4 |

---

## Trial 1 — stage1_baseline_and_profile [PASS]

**Metric**: 60.20 score  (verified FAIL)
**Elapsed**: 24m 26s
**Supervisor** (LLM): `continue` — Baseline established at score 60.20 with 74ms runtime. Profiling shows MIOpen LSTM dominates (99.28%). Multiple optimizations attempted in this trial. Advancing to optimization stage.
**Nudge agent**: 13 checks, 8 NOOPs, 5 nudge(s) sent
  - **13m 31s**: Writing a fully custom multi-layer LSTM in Triton is extremely unlikely to beat MIOpen's highly optimized `aten::miopen_rnn` which dominated your profile. Consider a more pragmatic approach: start from your already-passing baseline (score 60) that wraps `nn.LSTM`, and try `torch.compile` on the model or look for optimizations around the LSTM (e.g., fusing the h0/c0 initialization or reducing unnecessary computation since only `c_n` is returned). You've already spent significant time debugging correctness issues in a custom kernel that will likely be slower even when correct.
  - **16m 20s**: Don't settle for score 60. Try using half precision (`torch.float16` or `torch.bfloat16`) for the LSTM computation — the MI355X has much higher throughput for fp16/bf16 GEMM, and since GEMMs dominate 63% of your runtime, this could give a significant speedup. Cast weights and inputs to half precision, run the LSTM, then cast c_n back to float32 if needed for correctness checks.
  - **19m 17s**: Look at the reference `forward()` more carefully — it computes `self.fc(out[:, -1, :])` but discards the result, only returning `state[1]` (c_n). In your `ModelNew`, skip the `fc` computation entirely. Also, the reference calls `self.h0.to(x.device)` every forward pass — pre-move h0/c0 to CUDA in `__init__` or on first call and cache them to avoid repeated `.to()` overhead. These are small but free wins that compound over benchmark iterations.
  - **20m 52s**: Try wrapping the LSTM forward pass in a CUDA graph (`torch.cuda.CUDAGraph`) — with 6 layers × 512 timesteps, there are thousands of kernel launches, and CUDA graphs eliminate per-launch overhead. Do a warmup run first, then capture the graph, and replay it on each subsequent call. Make sure h0/c0 and the input tensor are allocated once and reused as the graph's input buffers.
  - **22m 19s**: Wrap the forward pass with `@torch.inference_mode()` decorator — this eliminates autograd tracking overhead (version counting, grad metadata) for every tensor operation, which adds up across 6 layers × 512 timesteps. This is a free speedup for inference-only workloads.

### Agent trajectory

1. `Used ReadFile (skills/amd-kernel-optimization/SKILL.md)`
2. `Used ReadFile (skills/gpu-profiling/SKILL.md)`
3. `Used ReadFile (.amdpilot/environment.md)`
4. `Used Shell (ls -la /workspace/)`
5. Let me read the test harness and the existing optimization state, then examine
6. `Used ReadFile (test_harness.py)`
7. `Used ReadFile (optimization_state.json)`
8. `Used Glob (**/37*.py)`
9. `Used Glob (KernelBench/**/37*.py)`
10. `Used Glob (KernelBench/**/*.py)`
11. Now let me read the problem 37 source file to understand the LSTM implementati
12. `Used ReadFile (KernelBench/KernelBench/level3/37_LSTMCn.py)`
13. Now I understand the problem. This is an LSTM model that:
14. `Used SetTodoList`
15. Run baseline benchmark to establish reference performance ←
16. Analyze LSTM math and understand the PyTorch reference implementation
17. Profile the reference implementation to identify bottlenecks
18. Design Triton kernel implementation for LSTM cell
19. Implement ModelNew class with Triton kernels
20. Verify correctness against PyTorch reference
21. Benchmark and optimize Triton implementation
22. Update optimization_state.json with results
23. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
24. Let me use the maximum timeout of 300 seconds:
25. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
26. The test harness expects a generated_kernel.py file. Since this is the baselin
27. `Used ReadFile (KernelBench/src/kernelbench/eval.py)`
28. Now I understand the evaluation flow. The harness:
29. `Used WriteFile (generated_kernel.py)`
30. Now let me run the baseline benchmark:
31. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
32. Excellent! I have the baseline:
33. Reference PyTorch LSTM: 74.1 ms
34. The baseline implementation matches exactly (speedup 1.0x)
35. Score: 60.0 (50 base + 10 for 1x speedup)
36. `Used SetTodoList`
37. Run baseline benchmark to establish reference performance
38. Analyze LSTM math and understand the PyTorch reference implementation ←
39. Profile the reference implementation to identify bottlenecks
40. Design Triton kernel implementation for LSTM cell
41. Implement ModelNew class with Triton kernels
42. Verify correctness against PyTorch reference
43. Benchmark and optimize Triton implementation
44. Update optimization_state.json with results
45. Now let me create a profiling script to understand the kernel breakdown:
46. `Used WriteFile (profile_lstm.py)`
47. `Used Shell (cd /workspace && /opt/ven...hon3 profile_lstm.py 2>&1)`
48. Excellent profiling data! Key insights:
49. Cijk_Alik_Bljk_SB_MT32x32x64_... (GEMM kernel): 62.9% of total time (180.3
50. LSTMFwdHidUpdate (LSTM forward hidden update): 35.8% of total time (102.6m
51. The LSTM cell computations (i, f, g, o gates) involve large matrix multiplic
52. For each of 6 layers × 512 sequence steps, we're doing 4 GEMMs (one per gate
53. The hidden state update kernel is also significant
54. `Used SetTodoList`
55. Run baseline benchmark to establish reference performance
56. Analyze LSTM math and understand the PyTorch reference implementation
57. Profile the reference implementation to identify bottlenecks
58. Design Triton kernel implementation for LSTM cell ←
59. Implement ModelNew class with Triton kernels
60. Verify correctness against PyTorch reference
61. Benchmark and optimize Triton implementation
62. Update optimization_state.json with results
63. Now let me study the LSTM computation more carefully. Looking at the PyTorch r
64. weight_ih shape: (4*hidden_size, input_size) - input to all 4 gates
65. weight_hh shape: (4*hidden_size, hidden_size) - hidden to all 4 gates
66. bias shape: (8*hidden_size,) - both input and hidden biases
67. `Used WriteFile (generated_kernel.py)`
68. Let me test this initial implementation:
69. `Used Shell (cd /workspace && /opt/ven... generated_kernel.py 2>&1)`
70. I see the issue - Triton doesn't support i_mask[None, ...] syntax. I need to f
71. `Used WriteFile (generated_kernel.py)`
72. Let me test this simpler implementation:
73. `Used Shell (cd /workspace && /opt/ven... generated_kernel.py 2>&1)`
74. I see the issue - my test is modifying x inside the loop. Let me fix the logic
75. `Used WriteFile (generated_kernel.py)`
76. `Used Shell (cd /workspace && /opt/ven... generated_kernel.py 2>&1)`
77. I see the bug - for layer > 0, the first timestep needs to use the layer's ini
78. `Used StrReplaceFile (generated_kernel.py)`
79. I need to restructure the code to properly handle multi-layer LSTM. Let me rew
80. `Used WriteFile (generated_kernel.py)`
81. `Used Shell (cd /workspace && /opt/ven... generated_kernel.py 2>&1)`
82. The implementation is correct! Now let me run the full benchmark with the actu
83. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
84. The test fails with the full problem parameters. The issue is that my test use
85. `Used WriteFile (check_init.py)`
86. `Used Shell (cd /workspace && /opt/ven...ython3 check_init.py 2>&1)`
87. I see - PyTorch initializes biases with the same uniform distribution as weigh
88. `Used Shell (/opt/venv/bin/python3 -c ...ourcefile(nn.LSTM))" 2>&1)`
89. `Used Shell (grep -A 30 "def reset_par...es/rnn.py | head -50 2>&1)`
90. I see! PyTorch initializes ALL parameters (including biases) with uniform(-std
91. `Used StrReplaceFile (generated_kernel.py)`
92. Now let me test again:
93. `Used Shell (cd /workspace && /opt/ven... generated_kernel.py 2>&1)`
94. Good, the small test still passes. Now let me run the full benchmark:
95. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
96. Still failing. The max difference is around 1.1 which suggests a significant n
97. `Used WriteFile (debug_bias.py)`
98. `Used Shell (cd /workspace && /opt/ven...ython3 debug_bias.py 2>&1)`
99. The bias layout looks correct. Let me check if there's an issue with how I'm l
100. `Used WriteFile (debug_compute.py)`
101. `Used Shell (cd /workspace && /opt/ven...on3 debug_compute.py 2>&1)`
102. Let me fix the debug script:
103. `Used StrReplaceFile (debug_compute.py)`
104. `Used Shell (cd /workspace && /opt/ven...on3 debug_compute.py 2>&1)`
105. The computation is correct. The issue must be in my Triton kernel. Let me look
106. `Used WriteFile (debug_strides.py)`
107. Good point from the supervisor. Let me take a step back and reconsider the app
108. `Used WriteFile (generated_kernel.py)`
109. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
110. The torch.compile approach gives correctness but no speedup (1.0x). This is ex
111. The model returns state[1] which is the final cell state c_n
112. The reference computes the full LSTM forward pass through all layers and tim
113. We only need the final cell state, not the hidden states or outputs
114. `Used WriteFile (generated_kernel.py)`
115. Wait, that won't work because the harness instantiates ModelNew directly. Let
116. `Used WriteFile (generated_kernel.py)`
117. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
118. That's worse - the first trials have high latency due to compilation overhead.
119. `Used WriteFile (generated_kernel.py)`
120. Now let me update the optimization state and finalize this stage. Given the pr
121. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
122. Good, we have a stable baseline with score 60. Now let me update the optimizat
123. `Used WriteFile (optimization_state.json)`
124. Excellent suggestion! The MI355X has much higher FP16/BF16 throughput. Let me
125. `Used WriteFile (generated_kernel.py)`
126. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
127. The FP16 approach has numerical differences (max diff ~0.0008, avg diff ~0.000
128. `Used Grep (get_tolerance_for_precision)`
129. The tolerance for FP32 is 1e-4 (0.0001). Our max diff of 0.0008 exceeds this.
130. `Used WriteFile (generated_kernel.py)`
131. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
132. Still 74ms. Let me try a different approach - use TF32 precision which might b
133. `Used WriteFile (generated_kernel.py)`
134. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
135. Still 74ms. The MIOpen LSTM implementation is extremely well optimized. Let me
136. `Used SetTodoList`
137. Run baseline benchmark to establish reference performance
138. Analyze LSTM math and understand the PyTorch reference implementation
139. Profile the reference implementation to identify bottlenecks
140. Design Triton kernel implementation for LSTM cell
141. Implement ModelNew class with Triton kernels
142. Verify correctness against PyTorch reference
143. Benchmark and optimize Triton implementation
144. Update optimization_state.json with results ←
145. `Used WriteFile (optimization_state.json)`
146. `Used SetTodoList`
147. Run baseline benchmark to establish reference performance
148. Analyze LSTM math and understand the PyTorch reference implementation
149. Profile the reference implementation to identify bottlenecks
150. Design Triton kernel implementation for LSTM cell
151. Implement ModelNew class with Triton kernels
152. Verify correctness against PyTorch reference
153. Benchmark and optimize Triton implementation
154. Update optimization_state.json with results
155. Let me run the final benchmark one more time to confirm our result:
156. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
157. I see some variance - the first run was 74.7ms but earlier runs were 74.0ms. T
158. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
159. Good catch! The reference computes fc but we don't need to since we only retur
160. `Used WriteFile (generated_kernel.py)`
161. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
162. There's some instability in the measurements (trials 51-70 show spikes). But l
163. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
164. Good idea! CUDA graphs can eliminate kernel launch overhead. Let me implement
165. `Used WriteFile (generated_kernel.py)`
166. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
167. The CUDA graph approach has issues with hipBLASLt during capture. This is a kn
168. `Used WriteFile (generated_kernel.py)`
169. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
170. There's some instability in the measurements (spikes at trials 63-69). The sec
171. `Used WriteFile (optimization_state.json)`
172. Good point! Let me add torch.inference_mode():
173. `Used WriteFile (generated_kernel.py)`
174. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
175. Still 74ms - the inference_mode doesn't help because the harness already runs
176. `Used WriteFile (optimization_state.json)`
177. The stage is complete. I've established a baseline of 74ms with score 60 for K
178. Custom Triton kernel: correctness issues
179. torch.compile: no speedup (MIOpen already optimal)
180. FP16: accuracy too low (0.0008 vs 0.0001 tolerance)
181. CUDA graphs: hipBLASLt doesn't support capture on ROCm
182. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 37 2>&1)`
183. **LLM Error**: LLM provider error: Error: peer closed connection without sending complete messa...

---
*Generated by amdpilot orchestrator*