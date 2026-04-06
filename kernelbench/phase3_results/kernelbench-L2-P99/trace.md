# amdpilot — kernelbench-L2-P99

| Property | Value |
|----------|-------|
| **Task** | Optimize KernelBench Level 2 Problem 99 with Triton on AMD MI355X |
| **Type** | optimize |
| **Duration** | 0.2h (875s) |
| **Best metric** | 60.10 score |
| **Trials** | 1 (0 failed) |
| **Stages** | 3 |

---

## Trial 1 — stage1_baseline_and_profile [PASS]

**Metric**: 60.10 score  (verified PASS)
**Elapsed**: 8m 26s
**Supervisor** (LLM): `continue` — Baseline established at score 60.10 (0.946ms vs 0.951ms reference, 1.005x speedup). Profiling shows GEMM at 98% and fused GELU+Softmax at 2%. Advancing to optimization stage.
**Nudge agent**: 7 checks, 6 NOOPs, 1 nudge(s) sent
  - **3m 28s**: The `unrealized_conversion_cast` LLVM error often occurs with `tl.dot` type mismatches on ROCm. For 8192×8192 matmul, rocBLAS (via `torch.mm`/`F.linear`) will be faster than any hand-written Triton matmul anyway. Split your approach: use `F.linear` for the matmul, then write a single Triton kernel that fuses only the GELU + Softmax (row-wise). This sidesteps the LLVM error and is likely the fastest strategy since the matmul is already highly optimized on MI355X.

### Agent trajectory

1. `Used ReadFile (skills/amd-kernel-optimization/SKILL.md)`
2. `Used ReadFile (skills/gpu-profiling/SKILL.md)`
3. `Used ReadFile (test_harness.py)`
4. `Used ReadFile (optimization_state.json)`
5. Let me check the existing structure and look for any previous implementation o
6. `Used Glob (**/*.py)`
7. `Used ReadFile (.amdpilot/environment.md)`
8. `Used Shell (ls -la /workspace/KernelBench/)`
9. `Used Shell (find /workspace/KernelBen...*" 2>/dev/null | head -20)`
10. `Used ReadFile (KernelBench/KernelBench/l...99_Matmul_GELU_Softmax.py)`
11. `Used ReadFile (KernelBench/runs/amdpilot...lem_99_sample_0_kernel.py)`
12. Now I understand the task. The previous failed attempt used tl.math.tanh which
13. Uses manual GELU implementation with tl.math.exp instead of tl.math.tanh
14. Follows ROCm Triton constraints (wavefront size 64, explicit float32 casts
15. `Used SetTodoList`
16. Run baseline test to establish PyTorch reference performance ←
17. Create corrected Triton implementation with manual GELU (no tl.math.tanh)
18. Verify correctness and benchmark the implementation
19. Profile to understand kernel breakdown
20. Update optimization_state.json with results
21. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
22. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
23. The test harness expects a generated_kernel.py file. Let me first create a wor
24. `Used WriteFile (generated_kernel.py)`
25. `Used TodoList ({"todos": "[{\"title\": \...status\": \"pending\"}]"})`
26. `Used SetTodoList`
27. Run baseline test to establish PyTorch reference performance
28. Create corrected Triton implementation with manual GELU (no tl.math.tanh)
29. Verify correctness and benchmark the implementation ←
30. Profile to understand kernel breakdown
31. Update optimization_state.json with results
32. Now let me run the benchmark to verify correctness and measure performance:
33. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
34. The kernel compiles and runs fast (~0.94ms vs the reference), but there's a co
35. `Used WriteFile (debug_kernel.py)`
36. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
37. There's a compilation error. The issue is with the .to(tl.float32) cast on the
38. `Used WriteFile (generated_kernel.py)`
39. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
40. The debug_kernel.py still has the old code. Let me update it:
41. `Used WriteFile (debug_kernel.py)`
42. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
43. The error is about LLVM translation failing for builtin.unrealized_conversion_
44. `Used Grep (@triton.jit)`
45. `Used Glob (KernelBench/results/**/*.py)`
46. `Used ReadFile (KernelBench/results/eval_...rnel_level_1_problem_1.py)`
47. I see - this working example uses num_warps and num_stages parameters. Let me
48. `Used WriteFile (generated_kernel.py)`
49. Good advice! Let me split the approach - use PyTorch's optimized F.linear for
50. `Used WriteFile (generated_kernel.py)`
51. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
52. Still getting correctness issues. The max difference is around 1.7, which is s
53. `Used WriteFile (debug_kernel.py)`
54. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
55. The small test works correctly with very small differences (~2e-5). The issue
56. `Used WriteFile (debug_kernel.py)`
57. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
58. Interesting! The max diff is 0.0005 which is small, but the test harness is re
59. `Used WriteFile (generated_kernel.py)`
60. `Used WriteFile (debug_kernel.py)`
61. `Used Shell (/opt/venv/bin/python3 /wo...pace/debug_kernel.py 2>&1)`
62. Now with BLOCK_SIZE=8192 (which is >= N), the results are correct with max dif
63. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
64. Excellent! The implementation is now correct and achieves a score of 60.1! The
65. Correctness: All 5 trials passed
66. Runtime: 0.946ms (new) vs 0.952ms (reference)
67. Speedup: 1.006x
68. `Used SetTodoList`
69. Run baseline test to establish PyTorch reference performance
70. Create corrected Triton implementation with manual GELU (no tl.math.tanh)
71. Verify correctness and benchmark the implementation
72. Profile to understand kernel breakdown ←
73. Update optimization_state.json with results
74. Now let me profile to understand the kernel breakdown. I'll create a profiling
75. `Used WriteFile (profile_kernel.py)`
76. `Used Shell (/opt/venv/bin/python3 /wo...ce/profile_kernel.py 2>&1)`
77. The profiling reveals important insights:
78. GEMM (rocBLAS): 97.5% (10.7ms)
79. Softmax: 1.5% (0.2ms)
80. GELU: 1.0% (0.1ms)
81. GEMM (rocBLAS): 98.1% (10.3ms)
82. gelu_softmax_kernel: 1.9% (0.2ms)
83. `Used ReadFile (optimization_state.json)`
84. `Used WriteFile (optimization_state.json)`
85. `Used SetTodoList`
86. Run baseline test to establish PyTorch reference performance
87. Create corrected Triton implementation with manual GELU (no tl.math.tanh)
88. Verify correctness and benchmark the implementation
89. Profile to understand kernel breakdown
90. Update optimization_state.json with results
91. Now let me run the final benchmark as required:
92. `Used Shell (/opt/venv/bin/python3 /wo...el 2 --problem-id 99 2>&1)`
93. The final benchmark shows:
94. SCORE: 60.1
95. Runtime: 0.946ms (new) vs 0.951ms (reference)
96. Speedup: 1.005x
97. All 5 correctness trials passed
98. `Used PyTorch's optimized F.linear (rocBLAS backend) for the matrix multiplic`
99. Created a fused Triton kernel for GELU + Softmax to reduce kernel launch ove
100. Implemented manual GELU using tl.math.exp since tl.math.tanh is unavailable
101. GEMM (rocBLAS): 98% of runtime (~10.3ms)
102. GELU+Softmax fused kernel: 2% of runtime (~0.2ms)
103. Correctness: ✓ All 5 trials passed
104. Performance: 0.946ms vs 0.951ms reference (1.005x speedup)
105. Score: 60.1
106. <choice>STOP</choice>

---
*Generated by amdpilot orchestrator*