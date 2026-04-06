# Learned Insights

- **Trial 1**: DenseNet201 consists of repeated DenseBlock layers (BN->ReLU->Conv2d->Dropout with growth_rate concatenation) and TransitionLayers (BN->ReLU->Conv2d->AvgPool2d), so fusing BN+ReLU is a high-value target across many layers.
- **Trial 2**: DenseNet201 consists of repeated DenseBlock layers (BN->ReLU->Conv2d->Dropout with growth_rate concatenation) and TransitionLayers (BN->ReLU->Conv2d->AvgPool2d), so fusing BN+ReLU is a high-value target across many layers.
- **Trial 2**: Agent has failed to produce any output in 2 trials - likely stuck on understanding the test harness structure. Must read test_harness.py first to understand expected file paths and class names.
- **Trial 2**: For KernelBench problems, the test harness expects a ModelNew class that is a drop-in replacement for Model. Start by aliasing ModelNew = Model to get a baseline score.
- **Trial 3**: Agent has failed to produce any output in 3 consecutive trials - likely a fundamental misunderstanding of the task setup or environment issue
- **Trial 3**: For KernelBench problems, the test harness expects a ModelNew class that is a drop-in replacement for Model. Start by aliasing ModelNew = Model to get a baseline score.
- **Trial 3**: CRITICAL: Always read test_harness.py first to understand the expected file paths, class names, and evaluation protocol before writing any code
- **Trial 4**: Agent has been completely stuck for 4 trials with zero output - needs extremely prescriptive step-by-step instructions
- **Trial 4**: CRITICAL: Always start by reading test_harness.py with cat to understand the expected file path and class interface before writing any code
- **Trial 4**: For KernelBench Level 3 Problem 16 (DenseNet201), start with ModelNew = Model as baseline to get a nonzero score before any optimization
- **Trial 5**: Agent has been completely stuck for 5 trials with zero output - needs literal copy-paste shell commands, not conceptual guidance
- **Trial 5**: CRITICAL: Always start with `cat /workspace/test_harness.py` as the very first command to understand the expected solution file path and class interface
- **Trial 5**: For KernelBench problems, create ModelNew as a trivial subclass of Model first to establish a nonzero baseline before any optimization
- **Trial 6**: Agent has been completely stuck for 6 trials with zero output - likely a fundamental execution environment issue or agent not receiving/processing instructions
- **Trial 6**: CRITICAL: Always start with `cat /workspace/test_harness.py` as the very first command
- **Trial 6**: For KernelBench problems, create ModelNew = Model as baseline to establish nonzero score before any optimization
- **Trial 7**: Agent has been completely non-functional for 7 trials producing zero output - likely a systemic execution issue rather than a guidance problem
- **Trial 7**: If agent produces zero output again after trial 8, escalate to ask_user to check environment
