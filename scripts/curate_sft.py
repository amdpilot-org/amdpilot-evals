#!/usr/bin/env python3
"""SFT data curation pipeline: internalize nudge agent signals into executor trajectories.

Two-phase approach:
  Phase 1 (regex): Mechanically identify nudge injection points (_steer calls +
  tool results) and remove them.  This is reliable structural detection.

  Phase 2 (LLM): For each nudge's executor response, call Claude opus via AMD
  Gateway to rewrite the thinking so it reads as if the executor independently
  arrived at the same conclusion.  Pattern matching is too brittle for the
  nuanced language variations in real acknowledgments.

  Phase 3 (LLM validation): After all rewrites, send representative chunks of
  the trajectory to Claude for a final check that zero traces remain.

Usage:
    # Dry run (regex-only, no LLM rewriting)
    python curate_sft.py --results-dir results/sglang-fused-moe-fix-run1 --dry-run

    # Full curation with Claude opus rewriting
    python curate_sft.py --results-dir results/sglang-fused-moe-fix-run1
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NUDGE_MARKER = "Supervisor Nudge #"
NUDGE_SYSTEM_PREFIX = "<system>The user has sent a real-time instruction:"

ACTION_PATTERNS = [
    r"run the benchmark", r"run .*bench", r"stop reading", r"stop exploring",
    r"act now", r"stop browsing", r"you must run", r"run it now", r"run `/",
    r"execute.*test", r"you haven't run", r"still haven't run",
    r"stop the package", r"benchmark output is truncated", r"tail.*bench.*log",
    r"reproduce the bug", r"reproduce.*first", r"check the crash",
    r"run the test", r"run.*test.harness", r"check.*error.output",
    r"check.*log", r"read the task", r"read.*requirements",
]

CORRECTION_PATTERNS = [
    r"revert", r"your .* is wrong", r"not the .* config",
    r"you're editing the wrong", r"crashing.*because your",
    r"regressed", r"worse than", r"you're getting.*error",
    r"that's not the.*bug", r"wrong file", r"wrong function",
    r"you're modifying.*wrong", r"that.*won't fix", r"not the root cause",
]

_AMD_GATEWAY_URL = "https://llm-api.amd.com/Anthropic"
_MODEL = "claude-opus-4-6"


# ===================================================================
# AMD Gateway LLM client
# ===================================================================

def _get_gateway_key() -> str:
    key = os.environ.get("LLM_GATEWAY_KEY", "")
    if not key:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("LLM_GATEWAY_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return key


def _call_opus(system_prompt: str, user_message: str, *, max_tokens: int = 4096) -> str:
    """Call Claude opus via AMD Gateway using raw HTTP (no SDK dependency)."""
    key = _get_gateway_key()
    if not key:
        raise RuntimeError("LLM_GATEWAY_KEY not found")

    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "amdpilot")

    payload = json.dumps({
        "model": _MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }).encode()

    req = urllib.request.Request(
        f"{_AMD_GATEWAY_URL}/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": key,
            "anthropic-version": "vertex-2023-10-16",
            "user": user,
        },
        method="POST",
    )

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        data = json.loads(resp.read().decode())

    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class NudgeInfo:
    index: int
    number: int
    text: str
    category: str
    response_index: int | None = None
    response_text: str = ""


@dataclass
class CurationReport:
    trial: int
    session_uuid: str
    total_lines: int
    nudges_found: int
    nudges_by_category: dict[str, int] = field(default_factory=dict)
    nudges: list[dict] = field(default_factory=list)
    lines_removed: int = 0
    lines_rewritten: int = 0
    output_lines: int = 0


# ===================================================================
# Phase 1: Mechanical detection (regex)
# ===================================================================

def classify_nudge(text: str) -> str:
    text_lower = text.lower()
    for pat in CORRECTION_PATTERNS:
        if re.search(pat, text_lower):
            return "correction"
    for pat in ACTION_PATTERNS:
        if re.search(pat, text_lower):
            return "action"
    return "direction"


def find_new_session(trial_dir: str, prev_trial_dir: str | None) -> str | None:
    def get_uuids(d: str) -> set[str]:
        uuids = set()
        for f in glob.glob(os.path.join(d, "sessions", "*", "*", "wire.jsonl")):
            uuids.add(os.path.basename(os.path.dirname(f)))
        return uuids

    current = get_uuids(trial_dir)
    previous = get_uuids(prev_trial_dir) if prev_trial_dir and os.path.isdir(prev_trial_dir) else set()

    new = current - previous
    if len(new) == 1:
        return new.pop()
    if len(new) == 0 and len(current) == 1:
        return current.pop()
    if len(new) > 1:
        log.warning("Multiple new sessions found: %s — using alphabetically last", new)
        return sorted(new)[-1]
    if len(current) > 0:
        return sorted(current)[-1]
    return None


def load_trial_context(trial_dir: str, session_uuid: str) -> list[dict]:
    session_dirs = glob.glob(os.path.join(trial_dir, "sessions", "*", session_uuid))
    if not session_dirs:
        return []
    session_dir = session_dirs[0]

    segments = []
    base = os.path.join(session_dir, "context.jsonl")
    if os.path.exists(base):
        segments.append(base)
    i = 1
    while True:
        seg = os.path.join(session_dir, f"context_{i}.jsonl")
        if not os.path.exists(seg):
            break
        segments.append(seg)
        i += 1

    lines = []
    for seg_path in segments:
        seg_name = os.path.basename(seg_path)
        with open(seg_path) as f:
            for line in f:
                entry = json.loads(line)
                entry["_source_file"] = seg_name
                lines.append(entry)
    return lines


def find_nudges(lines: list[dict]) -> list[NudgeInfo]:
    nudges = []
    for i, d in enumerate(lines):
        if d.get("role") != "tool":
            continue
        content = str(d.get("content", ""))
        if NUDGE_MARKER not in content:
            continue

        num_match = re.search(r"Supervisor Nudge #(\d+)", content)
        num = int(num_match.group(1)) if num_match else len(nudges) + 1

        nudge_text = content
        if NUDGE_SYSTEM_PREFIX in content:
            parts = content.split(NUDGE_SYSTEM_PREFIX, 1)
            nudge_text = parts[1].strip() if len(parts) > 1 else content
        header_end = nudge_text.find("\n\n")
        if header_end > 0:
            nudge_text = nudge_text[header_end:].strip()

        category = classify_nudge(nudge_text)

        resp_idx = None
        resp_text = ""
        for j in range(i + 1, min(i + 5, len(lines))):
            if lines[j].get("role") in ("_checkpoint", "_usage"):
                continue
            if lines[j].get("role") == "assistant":
                resp_idx = j
                c = lines[j].get("content", [])
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and "think" in part:
                            resp_text = part["think"]
                            break
                break

        nudges.append(NudgeInfo(
            index=i, number=num, text=nudge_text,
            category=category, response_index=resp_idx, response_text=resp_text,
        ))
    return nudges


def _find_steer_indices(lines: list[dict], nudge_index: int) -> list[int]:
    """Return indices of the synthetic _steer assistant message + metadata preceding a nudge."""
    to_remove = []
    for j in range(nudge_index - 1, max(nudge_index - 4, -1), -1):
        entry = lines[j]
        if entry.get("role") in ("_checkpoint", "_usage"):
            to_remove.append(j)
            continue
        if entry.get("role") == "assistant":
            raw = json.dumps(entry, ensure_ascii=False)
            if "_steer" in raw:
                to_remove.append(j)
            break
    return to_remove


def _find_trailing_metadata(lines: list[dict], nudge_index: int, exclude: set[int]) -> list[int]:
    """Return indices of _checkpoint/_usage entries immediately after the nudge."""
    to_remove = []
    for j in range(nudge_index + 1, min(nudge_index + 3, len(lines))):
        if j in exclude:
            continue
        if lines[j].get("role") in ("_checkpoint", "_usage"):
            to_remove.append(j)
        else:
            break
    return to_remove


# ===================================================================
# Phase 2: LLM-powered rewriting
# ===================================================================

_REWRITE_SYSTEM = """\
You are an expert editor for AI agent training data. Your task is to rewrite \
an AI coding agent's internal thinking (the <think> block) so that it reads \
as if the agent independently arrived at a conclusion, WITHOUT any external \
supervisor or nudge having told it what to do.

RULES:
1. The rewritten thinking must convey the SAME technical insight and lead to \
the SAME next action — you are only changing HOW the agent arrives at the idea, \
not WHAT it decides to do.
2. Remove ALL references to: supervisor, nudge, instruction, steer, guidance, \
"I was told", "I've been advised", "the suggestion", "as instructed", etc.
3. Replace acknowledgments with the agent's own reasoning. For example, \
"The supervisor says GEMM is the bottleneck" → "Looking at the profiling data, \
GEMM appears to be the bottleneck".
4. Match the original agent's language style, vocabulary, and level of detail. \
The agent uses a technical, first-person internal monologue style.
5. Keep the rewrite roughly the same length. Do not add explanations or \
commentary that wasn't in the original.
6. If the original thinking has substantive technical content AFTER the \
acknowledgment, preserve it exactly — only rewrite the acknowledgment portion.
7. Return ONLY the rewritten thinking text. No JSON, no markdown fences, \
no preamble."""

_VALIDATE_SYSTEM = """\
You are a quality auditor for AI training data. You are checking whether a \
trajectory has been properly cleaned of NUDGE signals — real-time steering \
injected by a monitoring agent during the trial.

IMPORTANT: The agent legitimately receives "supervisor hints" and "supervisor \
guidance" at the START of each trial from an inter-trial planning system. \
References to these are NORMAL and should NOT be flagged.

Only flag these NUDGE-specific issues:
1. References to "nudge", "_steer", "real-time instruction"
2. Messages containing "Supervisor Nudge #"
3. Acknowledgments like "The nudge says", "the steer told me"
4. Any text that implies the agent received mid-trial external steering \
   (as opposed to start-of-trial supervisor hints)

Do NOT flag:
- "supervisor guidance", "supervisor hints", "the supervisor instructed"
- These are legitimate inter-trial context, not nudge signals

Respond with a JSON object:
{"clean": true/false, "issues": ["issue description 1", ...]}
If clean, issues should be an empty list."""


def _llm_rewrite_think(
    original_think: str,
    nudge_text: str,
    category: str,
    context_before: str = "",
    full_nudge_tool_output: str = "",
) -> str:
    """Use Claude opus to rewrite an executor's thinking block.

    The LLM sees the full nudge as the executor received it (the tool result),
    the executor's thinking before and after, and category-specific guidance.
    This gives the LLM enough context to produce a natural rewrite that
    internalizes the nudge's insight.
    """
    cat_guidance = {
        "action": (
            "The nudge was a procedural prod (e.g. 'run the benchmark'). "
            "The agent's thinking starts by acknowledging it. Rewrite so the "
            "agent independently decides to take that action based on its own "
            "assessment of the situation."
        ),
        "direction": (
            "The nudge provided a domain insight (e.g. 'GEMM is the bottleneck'). "
            "Rewrite so the agent independently arrives at this insight through "
            "its own analysis of the data/code it has seen."
        ),
        "correction": (
            "The nudge corrected a mistake (e.g. 'revert, your change broke it'). "
            "Rewrite so the agent independently notices the mistake — perhaps by "
            "observing an error, noticing a regression, or re-reading the output."
        ),
    }

    user_msg = (
        f"## Nudge category: {category}\n"
        f"{cat_guidance.get(category, '')}\n\n"
    )
    if full_nudge_tool_output:
        user_msg += (
            f"## Full nudge as the executor received it (this tool result will be removed):\n"
            f"{full_nudge_tool_output[:800]}\n\n"
        )
    else:
        user_msg += (
            f"## The nudge that was sent (now being removed):\n"
            f"{nudge_text[:500]}\n\n"
        )
    if context_before:
        user_msg += (
            f"## What the agent was doing/thinking before the nudge:\n"
            f"{context_before[:600]}\n\n"
        )
    user_msg += (
        f"## Original thinking to rewrite (the agent's response after receiving the nudge):\n"
        f"{original_think[:2000]}\n\n"
        f"Rewrite the thinking above so it reads as independent reasoning. "
        f"The agent must arrive at the same conclusion and take the same action, "
        f"but through its OWN analysis — not because it was told. "
        f"Remove all acknowledgment of external guidance."
    )

    try:
        result = _call_opus(_REWRITE_SYSTEM, user_msg, max_tokens=2048)
        result = result.strip()
        if result:
            return result
    except Exception as e:
        log.warning("  LLM rewrite failed: %s — falling back to regex", e)

    return _regex_clean_think(original_think)


def _llm_validate_trajectory(curated: list[dict]) -> list[str]:
    """Use Claude opus to check for remaining nudge-specific traces (not supervisor hints)."""
    sample_thinks = []
    for d in curated:
        if d.get("role") != "assistant":
            continue
        c = d.get("content", [])
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and "think" in p:
                    think = p["think"]
                    if _has_nudge_reference(think):
                        sample_thinks.append(think[:300])

    if not sample_thinks:
        return []

    user_msg = (
        f"Check these {len(sample_thinks)} thinking blocks from a curated "
        f"agent trajectory for any remaining traces of external steering:\n\n"
    )
    for i, t in enumerate(sample_thinks[:20]):
        user_msg += f"--- Block {i+1} ---\n{t}\n\n"

    try:
        result = _call_opus(_VALIDATE_SYSTEM, user_msg, max_tokens=1024)
        data = json.loads(result)
        return data.get("issues", [])
    except Exception as e:
        log.warning("  LLM validation failed: %s", e)
        return []


# ===================================================================
# Regex fallback (used when --dry-run or LLM call fails)
# ===================================================================

def _regex_clean_think(think: str) -> str:
    """Regex-only fallback for cleaning acknowledgments."""
    cleaned = re.sub(
        r"(?i)(?:^|\.\s+|!\s+|\n)(?:"
        r"the supervisor[^.!]*[.!]|the nudge[^.!]*[.!]|supervisor nudge[^.!]*[.!]|"
        r"you're right[^.!]*[.!]|good point[^.!]*[.!]|real-time instruction[^.!]*[.!]|"
        r"as instructed[^.!]*[.!]|i've been told[^.!]*[.!]|let me follow[^.!]*[.!]|"
        r"was told to[^.!]*[.!]|been advised[^.!]*[.!]|supervisor's guidance[^.!]*[.!]"
        r")\s*",
        "",
        think,
    )
    cleaned = re.sub(
        r"(?i)^(the supervisor|the nudge|supervisor nudge|good point|you're right|"
        r"real-time instruction|as instructed|i've been told|let me follow|"
        r"was told to|been advised|supervisor's guidance)[^.!]*[.!]\s*",
        "",
        cleaned,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ===================================================================
# Nudge processing (combines regex detection + LLM rewriting)
# ===================================================================

def _get_context_before(lines: list[dict], nudge_index: int) -> str:
    """Extract the executor's thinking from the last assistant message before the nudge."""
    for j in range(nudge_index - 1, max(nudge_index - 10, -1), -1):
        if lines[j].get("role") == "assistant":
            raw = json.dumps(lines[j], ensure_ascii=False)
            if "_steer" in raw:
                continue
            c = lines[j].get("content", [])
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and "think" in p:
                        return p["think"][-500:]
            break
    return ""


def _rewrite_response_think(
    lines: list[dict],
    response_index: int,
    nudge: NudgeInfo,
    use_frontier: bool,
) -> None:
    """Rewrite the thinking in the executor's response to a nudge."""
    resp = lines[response_index]
    content = resp.get("content", [])
    if not isinstance(content, list):
        return

    context_before = _get_context_before(lines, nudge.index)

    full_nudge_output = ""
    nudge_entry = lines[nudge.index]
    raw_content = nudge_entry.get("content", "")
    if isinstance(raw_content, str):
        full_nudge_output = raw_content
    elif isinstance(raw_content, list):
        parts = []
        for p in raw_content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or p.get("think", ""))
            elif isinstance(p, str):
                parts.append(p)
        full_nudge_output = "\n".join(parts)

    new_content = []
    for part in content:
        if isinstance(part, dict) and "think" in part:
            original = part["think"]
            if use_frontier:
                rewritten = _llm_rewrite_think(
                    original, nudge.text, nudge.category,
                    context_before, full_nudge_output,
                )
            else:
                rewritten = _regex_clean_think(original)
            if not rewritten:
                rewritten = original
            new_content.append({**part, "think": rewritten})
        else:
            new_content.append(part)
    lines[response_index] = {**resp, "content": new_content}


def process_nudge(
    lines: list[dict],
    nudge: NudgeInfo,
    use_frontier: bool,
) -> set[int]:
    """Process a single nudge: remove injection, rewrite response."""
    to_remove = {nudge.index}
    to_remove.update(_find_steer_indices(lines, nudge.index))
    to_remove.update(_find_trailing_metadata(lines, nudge.index, to_remove))

    if nudge.response_index is not None:
        _rewrite_response_think(lines, nudge.response_index, nudge, use_frontier)

    if nudge.category in ("direction", "correction"):
        for j in range(nudge.index - 1, max(nudge.index - 10, -1), -1):
            if j in to_remove:
                continue
            if lines[j].get("role") == "assistant":
                raw = json.dumps(lines[j], ensure_ascii=False)
                if "_steer" in raw:
                    continue
                content = lines[j].get("content", [])
                if isinstance(content, list):
                    for k, part in enumerate(content):
                        if isinstance(part, dict) and "think" in part:
                            original = part["think"]
                            if len(original) > 50:
                                insight = nudge.text[:200].replace("\n", " ")
                                augmented = (
                                    original.rstrip()
                                    + f"\n\nLooking at this more carefully, {insight}"
                                )
                                content[k] = {**part, "think": augmented}
                                lines[j] = {**lines[j], "content": content}
                            break
                break

    return to_remove


_NUDGE_ONLY_KEYWORDS = [
    "nudge", "steer", "real-time instruction",
    "supervisor nudge", "the nudge",
]


def _has_nudge_reference(text: str) -> bool:
    """Check whether text references nudge-specific signals (NOT supervisor hints).

    Supervisor hints from retry_with_hints are LEGITIMATE inter-trial context
    that the executor should condition on.  Only the nudge agent's real-time
    _steer injections during a trial need to be cleaned.
    """
    lower = text.lower()
    return any(kw in lower for kw in _NUDGE_ONLY_KEYWORDS)


def _scrub_nudge_references(curated: list[dict]) -> None:
    """Remove residual nudge references from compaction summaries and any content."""
    _nudge_patterns = [
        re.compile(r"\*\*Supervisor [Nn]udge[s]?.*?\*\*.*?(?=\n\*\*|\n</|\n\n[A-Z<]|\Z)", re.DOTALL),
        re.compile(r"\*\*Supervisor [Nn]udge[s]? [Rr]eceived:?\*\*.*?(?=\n\*\*|\n</|\n\n[A-Z<]|\Z)", re.DOTALL),
        re.compile(r"\*\*Supervisor [Gg]uidance [Rr]eceived\*\*:?\s*\n(?:[-*] .*\n)*", re.MULTILINE),
        re.compile(r"(?:^|\n)[-*] *[Nn]udge #[\d,-]+:.*(?:\n|$)", re.MULTILINE),
        re.compile(r"[Ss]upervisor [Nn]udge[s]? (?:[Rr]eceived|[Ss]ummary):?\s*\n(?:[-*] .*\n)*", re.MULTILINE),
        re.compile(r"[Ss]upervisor [Gg]uidance [Rr]eceived:?\s*\n(?:[-*] .*\n)*", re.MULTILINE),
    ]
    for d in curated:
        raw = json.dumps(d, ensure_ascii=False)
        if "nudge" not in raw.lower() and NUDGE_MARKER not in raw:
            continue
        content = d.get("content")
        if isinstance(content, str):
            for pat in _nudge_patterns:
                content = pat.sub("", content)
            content = content.replace(NUDGE_MARKER, "")
            d["content"] = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    for key in ("text", "think"):
                        if key in part and isinstance(part[key], str):
                            text = part[key]
                            for pat in _nudge_patterns:
                                text = pat.sub("", text)
                            text = text.replace(NUDGE_MARKER, "")
                            part[key] = text


# ===================================================================
# Trial curation
# ===================================================================

def curate_trial(
    trial_dir: str,
    prev_trial_dir: str | None,
    trial_num: int,
    output_dir: str,
    *,
    use_frontier: bool = False,
) -> CurationReport | None:
    session_uuid = find_new_session(trial_dir, prev_trial_dir)
    if not session_uuid:
        log.warning("Trial %d: no session UUID found in %s", trial_num, trial_dir)
        return None

    lines = load_trial_context(trial_dir, session_uuid)
    if not lines:
        log.warning("Trial %d: no context lines for session %s", trial_num, session_uuid)
        return None

    log.info("Trial %d: session=%s, %d context lines", trial_num, session_uuid[:8], len(lines))

    nudges = find_nudges(lines)
    log.info("  Found %d nudges", len(nudges))

    report = CurationReport(
        trial=trial_num, session_uuid=session_uuid,
        total_lines=len(lines), nudges_found=len(nudges),
    )

    all_remove: set[int] = set()
    for nudge in nudges:
        log.info("    Nudge #%d [%s]: %s", nudge.number, nudge.category, nudge.text[:80])
        report.nudges.append({
            "number": nudge.number, "category": nudge.category,
            "index": nudge.index, "text": nudge.text[:300],
        })
        indices = process_nudge(lines, nudge, use_frontier)
        all_remove.update(indices)

    cat_counts: dict[str, int] = {}
    for n in nudges:
        cat_counts[n.category] = cat_counts.get(n.category, 0) + 1
    report.nudges_by_category = cat_counts
    report.lines_removed = len(all_remove)

    curated = []
    for i, d in enumerate(lines):
        if i in all_remove:
            continue
        clean = {k: v for k, v in d.items() if k != "_source_file"}
        curated.append(clean)

    _scrub_nudge_references(curated)

    # Phase 3: LLM validation — check for remaining nudge traces
    if use_frontier:
        issues = _llm_validate_trajectory(curated)
        if issues:
            log.warning("  LLM validation found %d issues:", len(issues))
            for issue in issues:
                log.warning("    - %s", issue)

    remaining = sum(
        1 for d in curated
        if NUDGE_MARKER in json.dumps(d, ensure_ascii=False)
    )
    if remaining > 0:
        log.warning("  WARNING: %d nudge markers still remain!", remaining)

    report.output_lines = len(curated)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"trial_{trial_num}_curated.jsonl")
    with open(out_path, "w") as f:
        for d in curated:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log.info("  Output: %s (%d lines, %d removed)", out_path, len(curated), len(all_remove))

    report_path = os.path.join(output_dir, f"trial_{trial_num}_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "trial": report.trial, "session_uuid": report.session_uuid,
            "total_lines": report.total_lines, "nudges_found": report.nudges_found,
            "nudges_by_category": report.nudges_by_category,
            "nudges": report.nudges, "lines_removed": report.lines_removed,
            "output_lines": report.output_lines,
            "remaining_nudge_traces": remaining,
        }, f, indent=2)

    return report


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Curate SFT data from amdpilot trajectories")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--trial", type=int, default=None, help="Process only this trial")
    parser.add_argument("--dry-run", action="store_true",
                        help="Regex-only mode, no LLM calls (fast but lower quality)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    agent_output = results_dir / "agent_output"
    if not agent_output.is_dir():
        log.error("No agent_output directory in %s", results_dir)
        sys.exit(1)

    use_frontier = not args.dry_run
    if use_frontier:
        key = _get_gateway_key()
        if not key:
            log.error("LLM_GATEWAY_KEY not found — cannot run LLM rewriting. Use --dry-run for regex-only mode.")
            sys.exit(1)
        log.info("Mode: LLM rewriting (Claude opus via AMD Gateway)")
    else:
        log.info("Mode: dry-run (regex-only, no LLM calls)")

    output_dir = args.output_dir or str(results_dir / "sft_curated")

    trial_dirs = sorted(glob.glob(str(agent_output / "trial_*_trajectory")))
    trials = []
    for td in trial_dirs:
        m = re.search(r"trial_(\d+)_trajectory", td)
        if m:
            trials.append((int(m.group(1)), td))

    if args.trial is not None:
        trials = [(n, d) for n, d in trials if n == args.trial]
        if not trials:
            log.error("Trial %d not found", args.trial)
            sys.exit(1)

    log.info("Processing %d trials from %s", len(trials), results_dir)
    log.info("Output: %s", output_dir)
    log.info("")

    all_reports = []
    for i, (trial_num, trial_dir) in enumerate(trials):
        prev_dir = trials[i - 1][1] if i > 0 else None
        report = curate_trial(
            trial_dir, prev_dir, trial_num, output_dir,
            use_frontier=use_frontier,
        )
        if report:
            all_reports.append(report)
        log.info("")

    summary_path = os.path.join(output_dir, "curation_summary.json")
    total_nudges = sum(r.nudges_found for r in all_reports)
    total_removed = sum(r.lines_removed for r in all_reports)
    cat_totals: dict[str, int] = {}
    for r in all_reports:
        for cat, cnt in r.nudges_by_category.items():
            cat_totals[cat] = cat_totals.get(cat, 0) + cnt

    summary = {
        "source": str(results_dir),
        "trials_processed": len(all_reports),
        "total_nudges": total_nudges,
        "total_lines_removed": total_removed,
        "nudges_by_category": cat_totals,
        "llm_rewriting": use_frontier,
        "per_trial": [
            {
                "trial": r.trial, "nudges": r.nudges_found,
                "categories": r.nudges_by_category,
                "lines_removed": r.lines_removed, "output_lines": r.output_lines,
            }
            for r in all_reports
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("CURATION SUMMARY")
    log.info("=" * 60)
    log.info("  Trials: %d", len(all_reports))
    log.info("  Total nudges: %d", total_nudges)
    log.info("  By category: %s", cat_totals)
    log.info("  Total lines removed: %d", total_removed)
    log.info("  LLM rewriting: %s", use_frontier)
    log.info("  Summary: %s", summary_path)


if __name__ == "__main__":
    main()
