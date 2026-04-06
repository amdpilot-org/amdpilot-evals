# ROCm: on_gfx*() arch detection crashes under torch.compile

The `on_gfx*()` architecture detection functions in the ROCm platform module use `@cache` combined with `amdsmi_init()` FFI calls at runtime. When these functions are invoked inside a `torch.compile()` region, TorchDynamo cannot trace the foreign function interface call and raises `torch._dynamo.exc.Unsupported`. This crashes any compiled code path that checks GPU architecture at runtime.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
