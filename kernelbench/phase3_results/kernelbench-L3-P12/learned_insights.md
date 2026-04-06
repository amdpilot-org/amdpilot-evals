# Learned Insights

- **Trial 1**: KernelBench scoring requires both correctness and speedup — a score of 0.00 means either no submission or correctness failure
- **Trial 1**: VGG19 has 16 Conv2d layers which dominate compute — fusing Conv2d+ReLU is the primary optimization target
- **Trial 2**: Agent produced no output in 2 consecutive trials — likely not finding or running the test harness correctly
- **Trial 2**: Must examine test_harness.py first to understand expected solution file location and format before writing any code
- **Trial 3**: Agent produced no output for 3 consecutive trials on KernelBench VGG19 — likely completely lost on environment setup
- **Trial 3**: Must start by reading test_harness.py to understand expected solution file path and class interface before writing any code
- **Trial 3**: A minimal pass-through ModelNew that copies the original Model is the critical first step to get a non-zero score
- **Trial 4**: Agent produced no output for 4 consecutive trials on KernelBench VGG19 — likely needs extremely prescriptive step-by-step instructions
- **Trial 4**: Critical first step: read test_harness.py to find expected solution file path, then create minimal ModelNew wrapping original Model
- **Trial 4**: torch.compile with mode='max-autotune' is the quickest path to speedup before writing custom Triton kernels
- **Trial 5**: Agent has been completely silent for 5 trials — likely failing at the very first step of finding/reading files
- **Trial 5**: Must provide exact shell commands for the agent to execute rather than high-level instructions
- **Trial 5**: The minimal viable solution is a ModelNew class that inherits from Model with no changes — this should produce a baseline score
- **Trial 6**: Agent has been completely silent for 6 trials — needs atomic step-by-step commands with explicit copy-paste instructions
- **Trial 6**: Must verify agent can execute basic shell commands before attempting any optimization
- **Trial 6**: The agent may be failing silently at startup — first command should be a simple cat/ls to verify execution
- **Trial 7**: Agent has been completely silent for 7 consecutive trials — may be a fundamental execution issue rather than a task comprehension issue
- **Trial 7**: If 8th trial also produces no output, should escalate to ask_user to check if the agent execution environment is functioning
