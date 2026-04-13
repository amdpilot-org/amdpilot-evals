#!/usr/bin/env python3
"""Fix circular import in aiter's fused_fp8_quant.py.

Problem: `fp8_dtype = aiter.dtypes.fp8` at module level fails because
aiter.__init__ hasn't finished setting attributes during the import chain.

Fix: Import dtypes directly from aiter.utility instead of through the
partially-initialized aiter module.
"""
import pathlib

target = pathlib.Path("/sgl-workspace/aiter/aiter/ops/triton/quant/fused_fp8_quant.py")
src = target.read_text()

src = src.replace(
    "fp8_dtype = aiter.dtypes.fp8",
    "from aiter.utility import dtypes as _aiter_dtypes\n"
    "fp8_dtype = _aiter_dtypes.fp8",
)

target.write_text(src)
print(f"Patched {target}")
