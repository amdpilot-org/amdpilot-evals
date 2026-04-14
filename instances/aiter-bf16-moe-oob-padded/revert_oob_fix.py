#!/usr/bin/env python3
"""Surgically revert the OOB bounds checks from CK commit 7eaedeb36.

This script modifies two CK header files to remove the bounds checks
that prevent OOB memory access on sentinel/padded MoE rows, restoring
the buggy behavior for eval purposes.
"""
import re

# 1. gridwise_moe_gemm.hpp: Replace the ternary bounds check with
#    unconditional weight access. The fix changed:
#      weight = p_sorted_weights_0[...]
#    to:
#      weight = token_offset < problem.NumTokens ? p_sorted_weights_0[...] : 0.0
#
#    We revert by replacing the multi-line ternary pattern back to
#    unconditional access.

gemm_path = (
    "/sgl-workspace/aiter/3rdparty/composable_kernel/"
    "include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp"
)

with open(gemm_path, "r") as f:
    content = f.read()

# The pattern appears twice (two GEMM stages). Each instance looks like:
#   return token_offset < problem.NumTokens
#       ? p_sorted_weights_0[IsInputGemm
#           ? token_offset
#           : token_offset * problem.TopK + (fused_token >> 24)]
#       : 0.0;
#
# Replace with unconditional access:
#   return p_sorted_weights_0[IsInputGemm
#       ? token_offset
#       : token_offset * problem.TopK + (fused_token >> 24)];

pattern = re.compile(
    r'return token_offset < problem\.NumTokens\s*'
    r'\?\s*(p_sorted_weights_0\[.*?\])\s*'
    r':\s*0\.0;',
    re.DOTALL
)

matches = list(pattern.finditer(content))
assert len(matches) == 2, f"Expected 2 OOB check instances, found {len(matches)}"

new_content = content
for m in reversed(matches):
    weight_expr = m.group(1)
    replacement = f"return {weight_expr};"
    new_content = new_content[:m.start()] + replacement + new_content[m.end():]

with open(gemm_path, "w") as f:
    f.write(new_content)

print(f"Reverted {len(matches)} OOB checks in gridwise_moe_gemm.hpp")

# 2. threadwise_tensor_slice_transfer_v7r3_scatter.hpp: Remove the
#    dst bounds check and always pass `true` to the scatter write.
#    The fix added:
#      const bool is_dst_valid = dst_offset < dst_descs[i].GetElementSpaceSize();
#    and changed the Update call from `true` to `is_dst_valid`.

scatter_path = (
    "/sgl-workspace/aiter/3rdparty/composable_kernel/"
    "include/ck/tensor_operation/gpu/thread/"
    "threadwise_tensor_slice_transfer_v7r3_scatter.hpp"
)

with open(scatter_path, "r") as f:
    scatter = f.read()

# Remove the is_dst_valid declaration line
scatter = scatter.replace(
    "const bool is_dst_valid = dst_offset < dst_descs[i].GetElementSpaceSize();\n",
    ""
)

# Replace is_dst_valid with true in the Update call
scatter = scatter.replace(
    "dst_offset, is_dst_valid,",
    "dst_offset, true,"
)

with open(scatter_path, "w") as f:
    f.write(scatter)

print("Reverted OOB check in threadwise_tensor_slice_transfer_v7r3_scatter.hpp")
