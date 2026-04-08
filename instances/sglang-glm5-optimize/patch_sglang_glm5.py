#!/usr/bin/env python3
"""Patch sglang for GLM-5 (glm_moe_dsa) compatibility.

Applied at Docker image build time. Two patches:
1. hf_transformers_utils.py: Add glm_moe_dsa fallback in AutoConfig error handler
2. model_config.py: Guard rope_scaling attribute access with getattr

GLM-5-FP8 uses model_type 'glm_moe_dsa' which is not in transformers <5.0.
transformers 5.x supports it natively but breaks DeepseekVL2Config dataclass
field ordering, so we pin to <5.0 and manually handle the config type.
"""
import sys

SGLANG_ROOT = "/sgl-workspace/sglang/python/sglang/srt"

# --- Patch 1: hf_transformers_utils.py ---
path1 = f"{SGLANG_ROOT}/utils/hf_transformers_utils.py"
with open(path1) as f:
    content = f.read()

old_handler = '        except ValueError as e:\n            if not "deepseek_v32" in str(e):\n                raise e'
new_handler = """        except ValueError as e:
            if "glm_moe_dsa" in str(e):
                config = PretrainedConfig.from_pretrained(
                    model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
                )
            elif not "deepseek_v32" in str(e):
                raise e"""

if old_handler in content:
    content = content.replace(old_handler, new_handler, 1)
    with open(path1, "w") as f:
        f.write(content)
    print(f"[PATCH 1] hf_transformers_utils.py: added glm_moe_dsa fallback")
elif "glm_moe_dsa" in content:
    print(f"[PATCH 1] hf_transformers_utils.py: already patched")
else:
    print(f"[PATCH 1] FAILED: could not find expected code pattern", file=sys.stderr)
    sys.exit(1)

# --- Patch 2: model_config.py ---
path2 = f"{SGLANG_ROOT}/configs/model_config.py"
with open(path2) as f:
    content2 = f.read()

changes = 0

# 2a: rope_scaling on hf_text_config
old_rope1 = "rope_scaling = self.hf_text_config.rope_scaling"
new_rope1 = 'rope_scaling = getattr(self.hf_text_config, "rope_scaling", None)'
if old_rope1 in content2:
    content2 = content2.replace(old_rope1, new_rope1)
    changes += 1

# 2b: rope_scaling on hf_config
old_rope2 = "if self.hf_config.rope_scaling:"
new_rope2 = 'if getattr(self.hf_config, "rope_scaling", None):'
if old_rope2 in content2:
    content2 = content2.replace(old_rope2, new_rope2)
    changes += 1

if changes > 0:
    with open(path2, "w") as f:
        f.write(content2)
    print(f"[PATCH 2] model_config.py: guarded {changes} rope_scaling access(es)")
elif 'getattr(self.hf_text_config, "rope_scaling"' in content2:
    print(f"[PATCH 2] model_config.py: already patched")
else:
    print(f"[PATCH 2] FAILED: could not find expected rope_scaling pattern", file=sys.stderr)
    sys.exit(1)

# --- Verify patches ---
print("\n--- Verification ---")
with open(path1) as f:
    c1 = f.read()
assert "glm_moe_dsa" in c1, "Patch 1 verification failed"
print("[OK] hf_transformers_utils.py has glm_moe_dsa handler")

with open(path2) as f:
    c2 = f.read()
assert 'getattr(self.hf_text_config, "rope_scaling"' in c2, "Patch 2a verification failed"
assert 'getattr(self.hf_config, "rope_scaling"' in c2, "Patch 2b verification failed"
print("[OK] model_config.py has guarded rope_scaling access")

# Syntax check both files
import py_compile
py_compile.compile(path1, doraise=True)
py_compile.compile(path2, doraise=True)
print("[OK] Both files pass syntax check")

print("\nAll patches applied and verified successfully.")
