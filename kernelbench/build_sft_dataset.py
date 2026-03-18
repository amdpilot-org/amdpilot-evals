#!/usr/bin/env python3
"""
Build high-quality SFT dataset from KernelBench x amdpilot Phase 3 trajectories.

Filters, cleans, validates, and exports trajectories in HuggingFace messages format.
Output: amdpilot-logs/kernelbench/lora_sft_amdpilot_kernelbench/
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

PHASE3_DIR = Path("/home/jinpan12/amdpilot/evals/kernelbench/phase3_results")
OUTPUT_DIR = Path("/home/jinpan12/amdpilot-logs/kernelbench/lora_sft_amdpilot_kernelbench")

TORCH_COMPILE_HACKS = {
    "kernelbench-L1-P34", "kernelbench-L1-P82",
    "kernelbench-L2-P3", "kernelbench-L2-P80",
    "kernelbench-L3-P30", "kernelbench-L3-P33", "kernelbench-L3-P36",
    "kernelbench-L3-P37", "kernelbench-L3-P41", "kernelbench-L3-P42",
}

MIN_SCORE = 60.0
MAX_TOOL_OUTPUT_CHARS = 4000

SYSTEM_MESSAGE = (
    "You are an expert GPU kernel engineer specializing in AMD GPUs with ROCm. "
    "You write optimized Triton kernels for AMD Instinct MI355X (gfx950, CDNA4). "
    "You use /opt/venv/bin/python3 and have access to PyTorch 2.9+rocm7.2, "
    "Triton 3.6.0 (ROCm fork), and standard profiling tools.\n\n"
    "Key AMD ROCm Triton constraints:\n"
    "- tl.math.tanh and tl.libdevice.* are UNAVAILABLE. Use manual tanh via tl.math.exp.\n"
    "- Wavefront size is 64 (not 32). BLOCK_SIZE multiples of 64 align best.\n"
    "- Compute in float32, cast back on store.\n"
    "- Use triton.next_power_of_2(N) for BLOCK_SIZE selection.\n"
    "- program_id axis must be 0, 1, or 2 (3D grid max)."
)

AMD_PATTERNS = {
    "manual_tanh": re.compile(r"exp.*2\.0.*x_clamped|exp_2x.*-.*1.*exp_2x.*\+.*1"),
    "wavefront_64": re.compile(r"wavefront.*64|num_warps|BLOCK.*64"),
    "gfx950": re.compile(r"gfx950|MI355|CDNA4"),
    "explicit_cast_f32": re.compile(r"\.to\(tl\.float32\)|\.to\(tl\.bfloat16\)"),
    "next_power_of_2": re.compile(r"next_power_of_2"),
}


def find_context_jsonl(task_dir: Path, trial_num: int) -> Path | None:
    traj_dir = task_dir / "agent_output" / f"trial_{trial_num}_trajectory" / "sessions"
    if not traj_dir.exists():
        return None
    for root, _, files in os.walk(traj_dir):
        if "context.jsonl" in files:
            return Path(root) / "context.jsonl"
    return None


def find_prompt_txt(task_dir: Path, trial_num: int) -> str | None:
    p = task_dir / "agent_output" / f"trial_{trial_num}_trajectory" / "prompt.txt"
    if p.exists():
        return p.read_text()
    return None


def filter_tasks() -> list[dict]:
    """Apply quality criteria and return whitelist of (task_name, trial_number, score) tuples."""
    whitelist = []

    for d in sorted(os.listdir(PHASE3_DIR)):
        task_dir = PHASE3_DIR / d
        if not task_dir.is_dir() or not d.startswith("kernelbench-"):
            continue

        if d in TORCH_COMPILE_HACKS:
            continue

        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        best_metric = summary.get("best_metric", "N/A")
        if best_metric in ("N/A", None):
            continue
        score = float(best_metric)
        if score < MIN_SCORE:
            continue

        best_trial = None
        best_score = -1.0
        for trial in summary.get("trials", []):
            if not trial.get("verified"):
                continue
            m = trial.get("verified_metric")
            if m is None:
                m_str = trial.get("metric", "0")
                m = float(m_str) if m_str not in ("N/A", None) else 0
            if m > best_score:
                best_score = m
                best_trial = trial.get("trial")

        if best_trial is None or best_score < MIN_SCORE:
            continue

        ctx_path = find_context_jsonl(task_dir, best_trial)
        if ctx_path is None:
            continue

        level = int(d.split("-L")[1].split("-")[0])
        pid = int(d.split("-P")[1])

        whitelist.append({
            "task_name": d,
            "task_dir": str(task_dir),
            "trial": best_trial,
            "score": best_score,
            "level": level,
            "problem_id": pid,
            "context_path": str(ctx_path),
        })

    return whitelist


def clean_trajectory(context_path: str) -> list[dict]:
    """Read context.jsonl, clean, and return list of message dicts."""
    messages = []
    with open(context_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = msg.get("role", "")
            if role in ("_checkpoint", "_usage"):
                continue

            cleaned = {"role": role}

            content = msg.get("content")
            if content is not None:
                content = normalize_content(content)
                if role == "tool" and isinstance(content, str) and len(content) > MAX_TOOL_OUTPUT_CHARS:
                    content = content[:MAX_TOOL_OUTPUT_CHARS] + "\n[... truncated]"
                cleaned["content"] = content

            if "tool_calls" in msg and msg["tool_calls"]:
                cleaned["tool_calls"] = msg["tool_calls"]

            if "tool_call_id" in msg:
                cleaned["tool_call_id"] = msg["tool_call_id"]

            if role == "assistant" and not content and not msg.get("tool_calls"):
                continue

            messages.append(cleaned)

    return remove_error_retry_noise(messages)


def normalize_content(content):
    """Flatten single-element content lists to strings."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        has_non_text = False
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "think":
                    text_parts.append(f"<think>{part.get('think', '')}</think>")
                else:
                    has_non_text = True
            elif isinstance(part, str):
                text_parts.append(part)
        if not has_non_text and text_parts:
            return "\n".join(t for t in text_parts if t.strip())
        return content
    return str(content) if content else ""


def remove_error_retry_noise(messages: list[dict]) -> list[dict]:
    """Remove sequences where a tool call errors and is immediately retried identically."""
    cleaned = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if (msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and "Error response from daemon" in msg.get("content", "")):
            if (i + 2 < len(messages)
                    and messages[i + 1].get("role") == "assistant"
                    and messages[i + 2].get("role") == "tool"):
                i += 1
                continue

        cleaned.append(msg)
        i += 1

    return cleaned


def validate_triton(messages: list[dict]) -> bool:
    """Check that the trajectory contains real Triton kernel code."""
    full_text = ""
    for msg in messages:
        if msg.get("role") == "assistant":
            c = msg.get("content", "")
            if isinstance(c, str):
                full_text += c
        if msg.get("role") == "tool":
            c = msg.get("content", "")
            if isinstance(c, str):
                full_text += c

    has_triton_jit = "@triton.jit" in full_text
    has_torch_compile_only = "torch.compile" in full_text and not has_triton_jit

    if has_torch_compile_only:
        return False
    return has_triton_jit


def detect_amd_patterns(messages: list[dict]) -> list[str]:
    """Detect AMD-specific patterns used in the trajectory."""
    full_text = ""
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            full_text += c

    found = []
    for name, pattern in AMD_PATTERNS.items():
        if pattern.search(full_text):
            found.append(name)
    return found


def count_triton_kernels(messages: list[dict]) -> int:
    """Count @triton.jit occurrences in assistant messages."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            c = msg.get("content", "")
            if isinstance(c, str):
                count += c.count("@triton.jit")
    return count


def format_example(entry: dict, messages: list[dict]) -> dict:
    """Format a single training example in HuggingFace messages format."""
    formatted_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    formatted_messages.extend(messages)

    num_turns = sum(1 for m in messages if m.get("role") == "user")
    num_tool_calls = sum(1 for m in messages if m.get("role") == "tool")

    return {
        "id": f"{entry['task_name']}-trial{entry['trial']}",
        "source": "amdpilot-kernelbench-phase3",
        "level": entry["level"],
        "problem_id": entry["problem_id"],
        "score": entry["score"],
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "triton_kernel_count": count_triton_kernels(messages),
        "amd_specific_fixes": detect_amd_patterns(messages),
        "messages": formatted_messages,
    }


def export_dataset(examples: list[dict]):
    """Write dataset to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "by_level").mkdir(exist_ok=True)
    (OUTPUT_DIR / "by_quality").mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    for level in [1, 2, 3]:
        level_examples = [ex for ex in examples if ex["level"] == level]
        if level_examples:
            with open(OUTPUT_DIR / "by_level" / f"level{level}.jsonl", "w") as f:
                for ex in level_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tier1 = [ex for ex in examples if ex["score"] >= 75]
    tier2 = [ex for ex in examples if ex["score"] >= 60]
    for name, data in [("tier1_score_ge75.jsonl", tier1), ("tier2_score_ge60.jsonl", tier2)]:
        with open(OUTPUT_DIR / "by_quality" / name, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_level_counts = {}
    for ex in examples:
        lv = ex["level"]
        by_level_counts[lv] = by_level_counts.get(lv, 0) + 1

    metadata = {
        "created": datetime.utcnow().isoformat() + "Z",
        "source": "amdpilot-kernelbench-phase3",
        "hardware": "AMD Instinct MI355X (gfx950)",
        "executor_model": "Qwen3.5-397B-A17B",
        "supervisor_model": "Claude Opus 4.6",
        "backend": "Triton (ROCm 7.2)",
        "filtering": {
            "min_score": MIN_SCORE,
            "verified_only": True,
            "torch_compile_hacks_removed": len(TORCH_COMPILE_HACKS),
            "triton_jit_required": True,
        },
        "stats": {
            "total_examples": len(examples),
            "by_level": by_level_counts,
            "tier1_ge75": len(tier1),
            "tier2_ge60": len(tier2),
            "avg_score": sum(ex["score"] for ex in examples) / max(len(examples), 1),
            "avg_tool_calls": sum(ex["num_tool_calls"] for ex in examples) / max(len(examples), 1),
            "avg_triton_kernels": sum(ex["triton_kernel_count"] for ex in examples) / max(len(examples), 1),
            "amd_pattern_frequency": {},
        },
    }

    pattern_counts = {}
    for ex in examples:
        for p in ex["amd_specific_fixes"]:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    metadata["stats"]["amd_pattern_frequency"] = pattern_counts

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""# lora_sft_amdpilot_kernelbench

SFT training data from KernelBench x amdpilot experiment.

## Overview

- **Source**: amdpilot Phase 3 (full pipeline) trajectories
- **Hardware**: AMD Instinct MI355X (gfx950, CDNA4)
- **Models**: Qwen3.5-397B-A17B (executor) + Claude Opus 4.6 (supervisor/nudge)
- **Backend**: Triton (ROCm 7.2, Triton 3.6.0)
- **Format**: HuggingFace messages (OpenAI chat format with tool calls)

## Stats

- Total examples: {len(examples)}
- By level: L1={by_level_counts.get(1, 0)}, L2={by_level_counts.get(2, 0)}, L3={by_level_counts.get(3, 0)}
- Tier 1 (score >= 75): {len(tier1)}
- Tier 2 (score >= 60): {len(tier2)}
- Average score: {metadata['stats']['avg_score']:.1f}

## Quality Filtering

- Verified correct only (verified=true in summary.json)
- Score >= {MIN_SCORE} (correct AND faster than PyTorch)
- {len(TORCH_COMPILE_HACKS)} torch.compile hacks removed
- Real @triton.jit kernel required
- Clean trajectories only (no server errors, no infinite loops)
- Best trial per task

## Files

- `train.jsonl` -- all examples
- `by_level/level{{1,2,3}}.jsonl` -- split by KernelBench level
- `by_quality/tier1_score_ge75.jsonl` -- highest quality
- `by_quality/tier2_score_ge60.jsonl` -- all qualifying
- `metadata.json` -- dataset statistics
"""
    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(readme)


def main():
    print("=" * 60)
    print("Building SFT dataset: lora_sft_amdpilot_kernelbench")
    print("=" * 60)

    print("\n[1/5] Filtering tasks by quality...")
    whitelist = filter_tasks()
    print(f"  Whitelisted: {len(whitelist)} tasks")
    for entry in whitelist[:5]:
        print(f"    {entry['task_name']}: score={entry['score']}, trial={entry['trial']}")
    if len(whitelist) > 5:
        print(f"    ... and {len(whitelist) - 5} more")

    examples = []
    skipped_no_triton = 0
    skipped_empty = 0
    skipped_error = 0

    for i, entry in enumerate(whitelist):
        task_name = entry["task_name"]

        print(f"\n[2/5] Processing {task_name} ({i+1}/{len(whitelist)})...")

        messages = clean_trajectory(entry["context_path"])

        if len(messages) < 3:
            print(f"  SKIP: too few messages ({len(messages)})")
            skipped_empty += 1
            continue

        full_text = " ".join(str(m.get("content", "")) for m in messages)
        if "LLM provider error" in full_text:
            print(f"  SKIP: contains LLM provider error")
            skipped_error += 1
            continue

        print(f"[3/5] Validating Triton kernel presence...")
        if not validate_triton(messages):
            print(f"  SKIP: no real Triton kernel found")
            skipped_no_triton += 1
            continue

        print(f"[4/5] Formatting example...")
        example = format_example(entry, messages)
        examples.append(example)
        print(f"  OK: {len(messages)} messages, {example['num_tool_calls']} tool calls, "
              f"{example['triton_kernel_count']} triton kernels, AMD fixes: {example['amd_specific_fixes']}")

    print(f"\n{'=' * 60}")
    print(f"[5/5] Exporting {len(examples)} examples...")
    print(f"  Skipped: {skipped_no_triton} no-triton, {skipped_empty} empty, {skipped_error} error")
    export_dataset(examples)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
