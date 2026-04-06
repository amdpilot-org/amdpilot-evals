# Bug: MoE Weight Shuffling Loses Custom Parameter Attributes

## Symptom

When unquantized MoE (Mixture of Experts) weight shuffling is performed in
the weight loading module, custom attributes that were
previously attached to `torch.nn.Parameter` objects (such as `weight_loader`)
are silently dropped.

This causes downstream failures in any code path that expects these attributes
to still be present on the parameters after shuffling. For example, model
checkpoint saving/loading pipelines that rely on `weight_loader` will break
because the attribute no longer exists on the shuffled parameter.

## How to Reproduce

1. Create a `torch.nn.Parameter` with a custom attribute (e.g.,
   `param.weight_loader = some_function`).
2. Assign it to an `nn.Module`.
3. Trigger the MoE weight shuffle code path.
4. Inspect the parameter afterwards -- the custom attribute is gone.

## Expected Behavior

Custom attributes on parameters should be preserved through the weight
shuffling process.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
