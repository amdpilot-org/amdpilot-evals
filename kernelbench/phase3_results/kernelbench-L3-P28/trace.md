# amdpilot — kernelbench-L3-P28

| Property | Value |
|----------|-------|
| **Task** | Optimize KernelBench Level 3 Problem 28 with Triton on AMD MI355X |
| **Type** | optimize |
| **Duration** | 0.2h (805s) |
| **Best metric** | 64.50 score |
| **Trials** | 1 (0 failed) |
| **Stages** | 3 |

---

## Trial 1 — stage1_baseline_and_profile [PASS]

**Metric**: 64.50 score  (verified PASS)
**Elapsed**: 10m 6s
**Supervisor** (LLM): `continue` — Baseline established at score 64.5 with profiling data: GEMM ~55%, Elementwise ~15%, Attention ~9.7%, LayerNorm ~9.3%. torch.compile already applied giving 1.5x speedup. Ready to advance to optimization stage.
**Nudge agent**: 5 checks, all NOOP (agent on track)

### Agent trajectory

1. `Used ReadFile (skills/amd-kernel-optimization/SKILL.md)`
2. `Used ReadFile (skills/gpu-profiling/SKILL.md)`
3. `Used ReadFile (test_harness.py)`
4. `Used ReadFile (optimization_state.json)`
5. Now I understand the task. I need to:
6. `Used Glob (/workspace/*.py)`
7. `Used Glob (/workspace/KernelBench/**/*.py)`
8. `Used Shell (ls -la /workspace/)`
9. Good - there's no generated_kernel.py yet. Let me first look at the KernelBenc
10. `Used Shell (ls -la /workspace/KernelBench/)`
11. Let me look at the KernelBench dataset to find problem 28:
12. `Used Shell (find /workspace/KernelBen...*" 2>/dev/null | head -20)`
13. `Used ReadFile (KernelBench/KernelBench/l...3/28_VisionTransformer.py)`
14. Now let me check if there's a previous attempt in the runs folder:
15. `Used ReadFile (KernelBench/runs/amdpilot...lem_28_sample_0_kernel.py)`
16. I can see the previous failed attempt. The key issue mentioned in the task was
17. `Used WriteFile (generated_kernel.py)`
18. Now let me run the baseline benchmark to establish the PyTorch reference perfo
19. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
20. Let me use the maximum timeout:
21. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
22. The baseline is established. The PyTorch reference runs at ~1.46ms and my curr
23. `Used WriteFile (profile_vit.py)`
24. `Used Shell (cd /workspace && /opt/ven...thon3 profile_vit.py 2>&1)`
25. Excellent! Now I have a clear picture of the bottlenecks:
26. Patch embedding linear: 768 -> 512
27. 6 transformer layers, each with:
28. QKV projection: 512 -> 1536 (3x512)
29. Output projection: 512 -> 512
30. MLP up: 512 -> 2048
31. MLP down: 2048 -> 512
32. MLP head: 512 -> 2048 -> 10
33. `Used WriteFile (generated_kernel.py)`
34. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
35. Excellent progress! torch.compile gave us a 1.41x speedup (1.58ms -> 1.12ms),
36. `Used WriteFile (generated_kernel.py)`
37. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
38. The output doesn't match because I changed the model structure (added batch_fi
39. `Used WriteFile (generated_kernel.py)`
40. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
41. Good - we're back to a working solution with 1.48x speedup (1.33ms -> 0.90ms),
42. `Used WriteFile (optimization_state.json)`
43. Now let me implement Triton kernels while keeping the exact same model archite
44. `Used WriteFile (generated_kernel.py)`
45. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
46. The Triton fused MLP kernel is producing slightly different results (the first
47. `Used WriteFile (generated_kernel.py)`
48. Let me run the benchmark again to confirm this works and get the final measure
49. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
50. We have a working solution with 1.45x speedup (1.31ms -> 0.90ms), score 64.5.
51. `Used WriteFile (generated_kernel.py)`
52. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
53. Good - we have a working solution with Triton kernels defined (for LayerNorm a
54. `Used WriteFile (optimization_state.json)`
55. Now let me run the final benchmark one more time to confirm the results:
56. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
57. Excellent! The final benchmark shows:
58. Runtime: 0.965ms (optimized)
59. Reference Runtime: 1.450ms (PyTorch baseline)
60. Speedup: 1.50x
61. SCORE: 65.0
62. `Used WriteFile (optimization_state.json)`
63. The Stage 1 baseline and profile task is complete. Here's a summary of what wa
64. PyTorch Reference Runtime: 1.45-1.46ms
65. Optimized Runtime: 0.965ms
66. Speedup: 1.50x
67. SCORE: 65.0
68. Manual tanh implementation (since tl.tanh is unavailable on ROCm)
69. Proper float32 casting
70. BLOCK_SIZE aligned to powers of 2
71. GEMM (Linear layers): ~55% of GPU time
72. Attention: ~9.7%
73. LayerNorm: ~9.3%
74. Elementwise ops: ~15%
75. Triton fused MLP kernel had numerical differences
76. Changing transformer to batch_first broke correctness
77. /workspace/generated_kernel.py - Contains the optimized ModelNew implementat
78. /workspace/optimization_state.json - Updated with profiling data and optimiz
79. `Used ReadFile (generated_kernel.py)`
80. `Used ReadFile (optimization_state.json)`
81. `Used Shell (cd /workspace && /opt/ven...el 3 --problem-id 28 2>&1)`
82. The benchmark completed successfully with:
83. Runtime: 0.878ms (optimized)
84. Reference Runtime: 1.280ms
85. Speedup: 1.46x
86. SCORE: 64.6
87. `Used STOP`
88. The task is complete. Final results:
89. Runtime: 0.878ms (optimized) vs 1.280ms (reference)
90. Speedup: 1.46x
91. SCORE: 64.6
92. Correctness: 5/5 trials passed
93. <choice>STOP</choice>

---
*Generated by amdpilot orchestrator*