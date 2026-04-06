# Learned Insights

- **Trial 1**: LLM provider connection errors can cause zero-work trials; always retry these.
- **Trial 2**: LLM provider connection errors can cause zero-work trials; always retry these.
- **Trial 2**: Two consecutive zero-output trials suggest the agent needs extremely detailed step-by-step instructions with exact commands.
- **Trial 3**: Three consecutive zero-output trials suggest persistent LLM provider connectivity issues or agent initialization failures.
- **Trial 3**: The agent needs extremely explicit copy-paste commands to overcome initialization barriers.
- **Trial 4**: Four consecutive zero-output trials suggest persistent agent initialization or LLM provider issues rather than task difficulty.
- **Trial 4**: For KernelBench problems, ModelNew must match the original Model's interface but can use optimized implementations.
- **Trial 4**: The simplest first optimization for ResNet18 is torch.compile wrapping the forward method.
- **Trial 5**: Five consecutive zero-output trials indicate a systemic agent or LLM provider issue, not task difficulty.
- **Trial 5**: For KernelBench ResNet18, ModelNew wraps Model and the simplest optimization is torch.compile(mode='max-autotune').
- **Trial 5**: The test harness expects get_inputs() and get_init_inputs() functions alongside ModelNew class in the problem file.
- **Trial 6**: Six consecutive zero-output trials indicate a systemic agent or LLM provider issue, not task difficulty.
- **Trial 6**: For KernelBench ResNet18, the minimum viable ModelNew is a subclass of Model with torch.compile wrapping forward.
- **Trial 6**: The test harness at /workspace/test_harness.py expects --level and --problem-id arguments.
- **Trial 6**: If torch.compile with max-autotune fails on AMD, try mode='default' or mode='reduce-overhead' instead.
- **Trial 7**: Seven consecutive zero-output trials indicate a systemic agent or LLM provider issue, not task difficulty.
- **Trial 7**: If agent consistently produces no output, consider asking user about infrastructure issues.
