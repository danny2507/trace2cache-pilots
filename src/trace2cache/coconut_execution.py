"""Coconut curriculum for stdout prediction.

Stage 0 is the existing event-chain LoRA (token CoT). Later stages replace
the first k compact event lines with the model's own last hidden states,
fed back as input embeddings. No tracer. No spliced event vectors.
"""

from __future__ import annotations

from trace2cache.execution_sim import (
    NATIVE_CAP,
    SFT_STDOUT,
    SimExample,
    gold_ask_events_text,
    sft_stdout_target,
    _mcnemar_pair,
)

COCONUT_K = 8
COCONUT_LATENT = "coconut_latent"
COCONUT_MIX = "coconut_mix"
DEFAULT_STAGES = (
    {"name": "mix8", "k": 8, "keep_events": True, "steps": 640},
    {"name": "latent8", "k": 8, "keep_events": False, "steps": 640},
)


def condition_for_stage(name: str) -> str:
    if name.startswith("mix"):
        return COCONUT_MIX
    if name.startswith("latent"):
        return COCONUT_LATENT
    raise ValueError(f"unknown Coconut stage {name}")


def copy_stdout_control_rows(rows: list[dict]) -> list[dict]:
    """Copy existing sft_stdout rows. Do not retrain the control."""
    copied = []
    for row in rows:
        if row.get("condition") != SFT_STDOUT:
            continue
        payload = dict(row)
        payload["thoughts"] = 0
        copied.append(payload)
    if not copied:
        raise ValueError("no sft_stdout control rows to copy")
    return copied


def gold_event_lines(example: SimExample, *, cap: int = NATIVE_CAP) -> tuple[str, ...]:
    if cap <= 0:
        raise ValueError("cap must be positive")
    text = gold_ask_events_text(example, cap=cap)
    return tuple(line for line in text.splitlines() if line.strip())


def coconut_language_target(
    example: SimExample,
    *,
    k: int,
    keep_events: bool,
    cap: int = NATIVE_CAP,
) -> str:
    """Tokens after the continuous thoughts. Loss is never on the thoughts."""
    if k < 1:
        raise ValueError("k must be positive")
    stdout = sft_stdout_target(example)
    if not keep_events:
        return stdout
    remaining = gold_event_lines(example, cap=cap)[k:]
    if not remaining:
        return stdout
    return "```\n" + "\n".join(remaining) + "\n```\n\n" + stdout


def stage_for_step(step: int, stages: tuple[dict, ...] = DEFAULT_STAGES) -> dict:
    if step < 1:
        raise ValueError("step must be positive")
    if not stages:
        raise ValueError("need at least one stage")
    cursor = 0
    for stage in stages:
        cursor += int(stage["steps"])
        if step <= cursor:
            return stage
    return stages[-1]


def total_steps(stages: tuple[dict, ...] = DEFAULT_STAGES) -> int:
    return sum(int(stage["steps"]) for stage in stages)


def summarize_coconut(rows: list[dict]) -> dict:
    """Kill rule: latent-only Coconut must beat stdout-only SFT on the frozen 128."""
    present = {row["condition"] for row in rows}
    required = {COCONUT_LATENT, SFT_STDOUT}
    if not required <= present:
        missing = sorted(required - present)
        raise ValueError(f"Coconut SFT needs {sorted(required)}; missing {missing}")
    conditions = [COCONUT_LATENT, SFT_STDOUT]
    if COCONUT_MIX in present:
        conditions = [COCONUT_LATENT, COCONUT_MIX, SFT_STDOUT]
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in conditions
    }
    n = {
        condition: sum(1 for row in rows if row["condition"] == condition)
        for condition in counts
    }
    latent_vs_stdout = _mcnemar_pair(rows, COCONUT_LATENT, SFT_STDOUT)
    beats = (
        latent_vs_stdout["left_only"] > latent_vs_stdout["right_only"]
        and latent_vs_stdout["p"] < 0.05
    )
    if beats:
        call = "coconut_pass"
        note = (
            "Coconut latent thoughts beat stdout-only SFT by McNemar. "
            "Continuous thoughts are live."
        )
    else:
        call = "coconut_fail"
        note = (
            "Coconut latent thoughts do not beat stdout-only SFT. "
            "Kill the method. Do not shuffle."
        )
    summary = {
        "call": call,
        "note": note,
        "n": n,
        "correct": counts,
        "latent_vs_stdout": latent_vs_stdout,
        "mean_thoughts": (
            sum(int(row.get("thoughts") or 0) for row in rows if row["condition"] == COCONUT_LATENT)
            / max(1, n[COCONUT_LATENT])
        ),
    }
    if COCONUT_MIX in present:
        summary["mix_vs_stdout"] = _mcnemar_pair(rows, COCONUT_MIX, SFT_STDOUT)
    return summary
