# Learned Insights

- **Trial 1**: LLM provider connection errors can cause zero-work trials — always retry these.
- **Trial 2**: LLM provider connection errors can cause zero-work trials — always retry these.
- **Trial 2**: Two consecutive zero-output trials suggest the agent needs extremely detailed step-by-step instructions to get started.
- **Trial 3**: Three consecutive zero-output trials suggest persistent LLM connectivity issues rather than task difficulty.
- **Trial 3**: Agent needs extremely explicit copy-paste commands to overcome initialization failures.
- **Trial 4**: Four consecutive zero-output trials suggest persistent agent/LLM connectivity issues rather than task difficulty.
- **Trial 4**: For KernelBench tasks, ModelNew must coexist with Model in the same file — do not delete Model.
- **Trial 4**: torch.compile with mode='max-autotune' is the simplest first optimization to try for KernelBench AlexNet.
- **Trial 5**: Five consecutive zero-output trials indicate persistent agent/LLM connectivity issues — providing complete copy-paste solutions is the only viable strategy.
- **Trial 5**: For KernelBench AlexNet, torch.compile with max-autotune and dropout=0.0 (eval mode) is the simplest optimization approach.
- **Trial 5**: ModelNew must be defined alongside Model in the same file for KernelBench test harness.
- **Trial 6**: Six consecutive zero-output trials indicate persistent agent/LLM connectivity issues — the agent may need a simpler, shorter prompt to succeed.
- **Trial 6**: For KernelBench AlexNet, torch.compile with max-autotune on feature extraction and classifier subgraphs separately can avoid graph breaks.
- **Trial 6**: Dropout with p=0.0 in eval mode is a no-op but the original Model includes it — ModelNew can omit it safely.
- **Trial 7**: Seven consecutive zero-output trials suggest the agent LLM provider has intermittent connectivity — keep prompts as short as possible.
- **Trial 7**: For KernelBench AlexNet, torch.compile with max-autotune on nn.Sequential feature/classifier blocks avoids graph breaks from dropout.
- **Trial 7**: ModelNew should omit Dropout layers entirely rather than using p=0.0 to simplify the graph for torch.compile.
