# Bug: dtype mismatch in gemm_a4w4 call within gemm_with_dynamic_quant

## Symptom

The `gemm_a4w4` kernel call inside `gemm_with_dynamic_quant` fails with a dtype mismatch error. The weight tensor is passed as `uint8`, but the kernel expects the weight tensor's dtype to match the quantized input tensor's dtype (e.g., `float8_e4m3fnuz`). This causes a runtime failure when the function attempts to perform the GEMM operation.

## How to reproduce

Run the test harness:

```
/opt/venv/bin/python3 /workspace/test_harness.py
```

The test constructs inputs that exercise the `gemm_with_dynamic_quant` code path and verifies that the GEMM completes without dtype errors.
