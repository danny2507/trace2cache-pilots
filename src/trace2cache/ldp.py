"""Latent Diagnose-then-Patch. Gate 1: can K slots hold an oracle REPLACE?

Encode input is the oracle sketch line only. No buggy source, no public test,
no trace. Shuffle is another task's REPLACE. Do not train Gate 2 on fail.
"""

from __future__ import annotations

import re

from trace2cache.execution_sim import _mcnemar_pair

LDP_K = 8
TRUE_LATENT = "true_latent"
SHUFFLED_LATENT = "shuffled_latent"
CORRUPTED_LATENT = "corrupted_latent"
NO_EVIDENCE = "no_evidence"
IO_TEXT = "io_text"
COMPACT_TEXT = "compact_text"
TEXT_CONTROLS = (NO_EVIDENCE, IO_TEXT, COMPACT_TEXT)
LATENT_CONDITIONS = (TRUE_LATENT, SHUFFLED_LATENT, CORRUPTED_LATENT)
_REPLACE_LINE = re.compile(r"REPLACE `(.*)` WITH `(.*)`")


def render_ldp_gate1_encode_text(sketch: str) -> str:
    """Encoder-only payload: the oracle REPLACE line. Nothing else."""
    text = sketch.strip()
    if not text:
        raise ValueError("empty oracle sketch")
    return text


def corrupt_edit_sketch(sketch: str) -> str:
    """Instance-specific corruption: swap REPLACE operands, else mark CORRUPTED."""
    text = sketch.strip()
    if not text or text == "NO_EDIT":
        return "CORRUPTED_SKETCH"
    lines = []
    for line in text.splitlines():
        match = _REPLACE_LINE.fullmatch(line)
        if match:
            swapped = f"REPLACE `{match.group(2)}` WITH `{match.group(1)}`"
            lines.append("CORRUPTED_SKETCH" if swapped == line else swapped)
            continue
        lines.append(line if line.startswith("CORRUPTED ") else f"CORRUPTED {line}")
    corrupted = "\n".join(lines)
    if corrupted == text:
        return "CORRUPTED_SKETCH"
    return corrupted


def copy_text_control_rows(rows: list[dict]) -> list[dict]:
    """Copy existing no_evidence / io_text / compact_text rows. Do not retrain them."""
    copied = [dict(row) for row in rows if row.get("condition") in TEXT_CONTROLS]
    present = {row["condition"] for row in copied}
    missing = [condition for condition in TEXT_CONTROLS if condition not in present]
    if missing:
        raise ValueError(f"no text control rows to copy: missing {missing}")
    return copied


def summarize_ldp_gate1(rows: list[dict]) -> dict:
    """Kill: true_latent must beat no_evidence by McNemar p<0.05 and shuffle must drop."""
    present = {row["condition"] for row in rows}
    required = {TRUE_LATENT, SHUFFLED_LATENT, NO_EVIDENCE}
    if not required <= present:
        missing = sorted(required - present)
        raise ValueError(f"LDP Gate 1 needs {sorted(required)}; missing {missing}")
    conditions = [TRUE_LATENT, SHUFFLED_LATENT, NO_EVIDENCE]
    if CORRUPTED_LATENT in present:
        conditions.append(CORRUPTED_LATENT)
    if IO_TEXT in present:
        conditions.append(IO_TEXT)
    if COMPACT_TEXT in present:
        conditions.append(COMPACT_TEXT)
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in conditions
    }
    n = {
        condition: sum(1 for row in rows if row["condition"] == condition)
        for condition in counts
    }
    vs_no_evidence = _mcnemar_pair(rows, TRUE_LATENT, NO_EVIDENCE)
    vs_shuffled = _mcnemar_pair(rows, TRUE_LATENT, SHUFFLED_LATENT)
    beats_no_evidence = (
        vs_no_evidence["left_only"] > vs_no_evidence["right_only"]
        and vs_no_evidence["p"] < 0.05
    )
    shuffle_drop = vs_shuffled["left_only"] > vs_shuffled["right_only"]
    if beats_no_evidence and shuffle_drop:
        call = "ldp_g1_pass"
        note = (
            "K=8 slots hold an oracle REPLACE and shuffle dropped. "
            "Channel is sufficient. Go to Gate 2."
        )
    else:
        call = "ldp_g1_fail"
        note = (
            "K=8 inputs_embeds cannot carry this REPLACE. "
            "Kill Gate 1. Do not train Gate 2."
        )
    summary = {
        "call": call,
        "note": note,
        "correct": counts,
        "n": n,
        "true_vs_no_evidence": vs_no_evidence,
        "true_vs_shuffled": vs_shuffled,
        "beats_no_evidence": beats_no_evidence,
        "shuffle_drop": shuffle_drop,
    }
    if CORRUPTED_LATENT in present:
        summary["true_vs_corrupted"] = _mcnemar_pair(rows, TRUE_LATENT, CORRUPTED_LATENT)
    if COMPACT_TEXT in present:
        summary["true_vs_compact"] = _mcnemar_pair(rows, TRUE_LATENT, COMPACT_TEXT)
    return summary
