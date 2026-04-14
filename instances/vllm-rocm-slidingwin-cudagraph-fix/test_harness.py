#!/usr/bin/env python3
"""Test harness for vllm PR #36042: CUDA graph decode capture crash in AITER FA.

Bug: The decode path in AITER FlashAttention dispatches to unified_attention
for sliding window models even during normal single-token decode. The
unified_attention Triton kernel path is incompatible with CUDA graph capture,
causing a crash when the engine tries to capture the decode graph. The dispatch
condition should only trigger for multi-token decode scenarios (e.g.,
speculative decoding), not for sliding window attention.

Tests:
  1. Dispatch condition for unified_attention does not reference sliding_window.
  2. Assert message references the correct reason for the dispatch.
  3. No sliding_window in the unified_attention dispatch condition.
  4. Dispatch condition is not a compound condition with unrelated checks.
"""
import ast
import os
import sys

checks_passed = 0
checks_total = 0

AITER_FA_PATH = "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_fa.py"


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


print("=" * 60)
print("vllm-rocm-slidingwin-cudagraph-fix test harness (PR #36042)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm_aiter_fa.py exists", os.path.isfile(AITER_FA_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

try:
    with open(AITER_FA_PATH) as f:
        source = f.read()
    tree = ast.parse(source)
except SyntaxError as e:
    check("rocm_aiter_fa.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source_lines = source.splitlines()

# ---------------------------------------------------------------------------
# Find the unified_attention dispatch block.
#
# We need to locate the if-condition that routes to unified_attention
# in the decode path. Before the fix this was:
#   if self.sliding_window[0] != -1 or decode_max_query_len > 1:
# After the fix:
#   if decode_max_query_len > 1:
# ---------------------------------------------------------------------------

# Find all If nodes that reference unified_attention in their body
# (specifically the dispatch condition, not the head_size fallback from #34570)

class UnifiedAttentionDispatchFinder(ast.NodeVisitor):
    """Find the if-condition that dispatches to unified_attention
    for speculative decoding (max_query_len > 1)."""

    def __init__(self, source_text):
        self.source_text = source_text
        self.source_lines = source_text.splitlines()
        self.dispatch_conditions = []  # (line_no, condition_source, body_source)

    def visit_If(self, node):
        # Get the source for this if-block's body
        body_start = node.body[0].lineno if node.body else node.lineno
        body_end = node.body[-1].end_lineno if node.body else node.lineno
        body_source = "\n".join(self.source_lines[body_start - 1:body_end])

        # Check if the body contains unified_attention usage (import or call)
        if "unified_attention" in body_source and "import" not in self.source_lines[node.lineno - 1]:
            # Get the condition source
            cond_source = "\n".join(self.source_lines[node.lineno - 1:node.lineno])
            # Also check for max_query_len to distinguish from head_size fallback
            if "query_len" in cond_source or "max_query_len" in cond_source:
                self.dispatch_conditions.append({
                    "line": node.lineno,
                    "condition": cond_source.strip(),
                    "body": body_source,
                })

        self.generic_visit(node)


finder = UnifiedAttentionDispatchFinder(source)
finder.visit(tree)

# ---------------------------------------------------------------------------
# Check 1: Dispatch condition references max_query_len > 1 but NOT
# sliding_window.
#
# Before fix: `if self.sliding_window[0] != -1 or decode_max_query_len > 1:`
# After fix: `if decode_max_query_len > 1:`
# ---------------------------------------------------------------------------
print("\n--- Check 1: Dispatch condition ---")

has_query_len_dispatch = len(finder.dispatch_conditions) > 0
has_sliding_window_in_cond = False

for dc in finder.dispatch_conditions:
    if "sliding_window" in dc["condition"]:
        has_sliding_window_in_cond = True

check(
    "Dispatch condition for unified_attention exists with max_query_len",
    has_query_len_dispatch,
    "No unified_attention dispatch condition with max_query_len found",
)

check(
    "Dispatch condition does NOT reference sliding_window",
    has_query_len_dispatch and not has_sliding_window_in_cond,
    "sliding_window still in dispatch condition — will crash CUDA graph capture for sliding window models",
)

# ---------------------------------------------------------------------------
# Check 2: Assert message mentions speculative decoding only.
#
# Before fix: "Shuffle KV cache layout is not supported with sliding
#              window or speculative decoding (multi-token decode)."
# After fix:  "Shuffle KV cache layout is not supported with
#              speculative decoding (multi-token decode)."
# ---------------------------------------------------------------------------
print("\n--- Check 2: Assert message ---")

# Find assert statements near the unified_attention dispatch
has_spec_decode_assert = False
assert_mentions_sliding = False

for dc in finder.dispatch_conditions:
    body = dc["body"]
    if "assert" in body:
        # Extract just the assert message string(s), not the entire body
        # The assert message is typically in string literals after "assert ..."
        body_lines = body.splitlines()
        in_assert = False
        assert_text = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith("assert "):
                in_assert = True
                assert_text.append(stripped)
            elif in_assert and (stripped.startswith('"') or stripped.startswith("'")):
                assert_text.append(stripped)
            elif in_assert and stripped.endswith(")"):
                assert_text.append(stripped)
                in_assert = False
            elif in_assert:
                assert_text.append(stripped)
        assert_msg = " ".join(assert_text).lower()
        if "speculative" in assert_msg or "multi-token" in assert_msg:
            has_spec_decode_assert = True
        if "sliding" in assert_msg:
            assert_mentions_sliding = True

check(
    "Assert message mentions speculative decoding",
    has_spec_decode_assert,
    "Assert message should reference speculative/multi-token decoding as the reason",
)

check(
    "Assert message does NOT mention sliding window",
    has_spec_decode_assert and not assert_mentions_sliding,
    "Assert still mentions sliding window — should only reference speculative decoding",
)

# ---------------------------------------------------------------------------
# Check 3: No sliding_window reference in the dispatch CONDITION.
#
# The fix removes the sliding_window condition from the if-statement.
# Note: sliding_window may still appear in the body as a function argument
# (e.g., window_size=self.sliding_window) — that's expected and correct.
# ---------------------------------------------------------------------------
print("\n--- Check 3: Clean dispatch condition ---")

sliding_in_condition = False
for dc in finder.dispatch_conditions:
    if "sliding_window" in dc["condition"]:
        sliding_in_condition = True

check(
    "No sliding_window in dispatch condition (may still be in function args)",
    has_query_len_dispatch and not sliding_in_condition,
    "sliding_window still in dispatch condition — will crash CUDA graph capture",
)

# ---------------------------------------------------------------------------
# Check 4: The dispatch condition is ONLY max_query_len > 1.
#
# Verify the condition is a simple comparison, not a compound condition
# with other checks (like sliding_window).
# ---------------------------------------------------------------------------
print("\n--- Check 4: Simple dispatch condition ---")

is_simple_condition = False
for dc in finder.dispatch_conditions:
    cond = dc["condition"]
    # Should be something like: "if decode_max_query_len > 1:"
    # Should NOT contain "or" (which would indicate compound condition)
    if "query_len" in cond and " or " not in cond:
        is_simple_condition = True

check(
    "Dispatch condition is simple (no compound 'or' with other checks)",
    is_simple_condition,
    "Dispatch condition still has compound 'or' — should only check max_query_len > 1",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
