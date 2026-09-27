"""ReflectionCoder-style sketch distillation for MBPP repair.

Same no_evidence prompt for both LoRAs. Distill target is the oracle REPLACE
then a fenced canonical patch. Eval never sees the oracle. Direct SFT is the
ablation: fenced patch only.
"""

from __future__ import annotations

from trace2cache.execution_sim import _mcnemar_pair
from trace2cache.mbpp_generalization import RepairExample, edit_sketch, render_chat

DIRECT_SFT = "direct_sft"
SKETCH_DISTILL = "sketch_distill"
NO_EVIDENCE = "no_evidence"
IO_TEXT = "io_text"
COMPACT_TEXT = "compact_text"
TEXT_CONTROLS = (NO_EVIDENCE, IO_TEXT, COMPACT_TEXT)
TRAIN_OBJECTIVES = (DIRECT_SFT, SKETCH_DISTILL)
REPAIR_EVIDENCE = "unavailable"


def fenced_patch(source: str) -> str:
    return f"```python\n{source.rstrip()}\n```"


def sketch_prefix(example: RepairExample) -> str:
    return edit_sketch(example.buggy_source, example.target_source) + "\n\n"


def sft_target(example: RepairExample, objective: str) -> str:
    fence = fenced_patch(example.target_source)
    if objective == DIRECT_SFT:
        return fence
    if objective == SKETCH_DISTILL:
        return sketch_prefix(example) + fence
    raise ValueError(f"unknown objective: {objective}")


def render_repair_chat(
    tokenizer,
    example: RepairExample,
    *,
    include_specification: bool = False,
    include_public_test: bool = True,
) -> str:
    """Same prompt as frozen no_evidence: public test on, spec off, evidence unavailable."""
    return render_chat(
        tokenizer,
        example,
        REPAIR_EVIDENCE,
        include_specification=include_specification,
        include_public_test=include_public_test,
        evidence_kind="runtime",
    )


def prompt_leaks_oracle(prompt: str, example: RepairExample) -> bool:
    sketch = edit_sketch(example.buggy_source, example.target_source)
    if sketch != "NO_EDIT" and sketch in prompt:
        return True
    canonical = example.target_source.strip()
    return bool(canonical) and canonical in prompt


def copy_text_control_rows(rows: list[dict]) -> list[dict]:
    """Copy existing no_evidence / io_text / compact_text rows. Do not retrain them."""
    copied = [dict(row) for row in rows if row.get("condition") in TEXT_CONTROLS]
    if not any(row.get("condition") == NO_EVIDENCE for row in copied):
        raise ValueError("no no_evidence rows to copy")
    return copied


def summarize_sketch_distill(rows: list[dict]) -> dict:
    """Pass iff sketch_distill beats direct_sft and no_evidence by McNemar p<0.05."""
    present = {row["condition"] for row in rows}
    required = {SKETCH_DISTILL, DIRECT_SFT, NO_EVIDENCE}
    if not required <= present:
        missing = sorted(required - present)
        raise ValueError(f"sketch distill needs {sorted(required)}; missing {missing}")
    conditions = [SKETCH_DISTILL, DIRECT_SFT, NO_EVIDENCE]
    for extra in (IO_TEXT, COMPACT_TEXT):
        if extra in present:
            conditions.append(extra)
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in conditions
    }
    n = {
        condition: sum(1 for row in rows if row["condition"] == condition)
        for condition in counts
    }
    vs_direct = _mcnemar_pair(rows, SKETCH_DISTILL, DIRECT_SFT)
    vs_no_evidence = _mcnemar_pair(rows, SKETCH_DISTILL, NO_EVIDENCE)
    beats_direct = vs_direct["left_only"] > vs_direct["right_only"] and vs_direct["p"] < 0.05
    beats_no_evidence = (
        vs_no_evidence["left_only"] > vs_no_evidence["right_only"] and vs_no_evidence["p"] < 0.05
    )
    if beats_direct and beats_no_evidence:
        call = "sketch_distill_pass"
        note = (
            "Sketch distillation beats direct repair LoRA and no_evidence by McNemar. "
            "Keep the method."
        )
    else:
        call = "sketch_distill_fail"
        note = (
            "Sketch distillation did not beat both direct_sft and no_evidence. "
            "Kill the method. Do not train it again."
        )
    return {
        "call": call,
        "note": note,
        "correct": counts,
        "n": n,
        "sketch_vs_direct": vs_direct,
        "sketch_vs_no_evidence": vs_no_evidence,
        "beats_direct": beats_direct,
        "beats_no_evidence": beats_no_evidence,
    }
