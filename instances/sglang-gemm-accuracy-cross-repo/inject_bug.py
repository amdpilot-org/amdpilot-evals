#!/usr/bin/env python3
"""Inject MoE output scaling bug into aiter's fused_moe.py."""

path = "/sgl-workspace/aiter/aiter/fused_moe.py"
with open(path) as f:
    content = f.read()

# Insert moe_buf.mul_(0.01) before the final return in fused_moe_sub.
# This scales down MoE expert outputs by 100x, causing the residual
# connection to dominate and producing degraded model outputs.
old = "    return moe_buf\n\n\n@functools.lru_cache(maxsize=2048)\ndef get_block_size_M"
new = "    moe_buf.mul_(0.01)\n    return moe_buf\n\n\n@functools.lru_cache(maxsize=2048)\ndef get_block_size_M"

assert old in content, f"Pattern not found in {path}"
content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print("Bug injected successfully")
