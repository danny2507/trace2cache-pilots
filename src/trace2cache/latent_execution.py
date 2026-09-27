"""Trace-grounded latent execution: distill interpreter events into a LoRA rollout.

The frozen splice never learned to read event embeddings. This module unfreezes a
LoRA on the same 3B and teacher-forces resampled native event vectors, with a
projector that must write those vectors from hidden states. At test the tracer is
gone: code + stdin + K projected states → stdout.

Shuffle trains on another task's events. Do not hide stdin. Do not train on the
frozen 128.
"""

from __future__ import annotations

from .execution_sim import (
    native_event_texts,
    render_chat,
    evidence_for_condition,
)
from .mbpp_generalization import EvidenceEvent

DISTILL_K = 16
OBJECTIVES = ("code_sft", "latent", "shuffled")
LATENT_EVAL_CONDITIONS = ("code", "rollout", "teacher")


def resample_indices(n: int, k: int) -> tuple[int, ...]:
    """Uniform index map from n teacher events onto k latent slots."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if k == 1:
        return (0,)
    return tuple(int(round(i * (n - 1) / (k - 1))) for i in range(k))


def resample_events(
    events: tuple[EvidenceEvent, ...], k: int
) -> tuple[EvidenceEvent, ...]:
    if not events:
        raise ValueError("need at least one event to resample")
    return tuple(events[i] for i in resample_indices(len(events), k))


def teacher_texts(events: tuple[EvidenceEvent, ...], k: int) -> tuple[str, ...]:
    return native_event_texts(resample_events(events, k), cap=k)


def distill_prompt(tokenizer, example) -> str:
    """Always the code+stdin prompt. Traces never enter the text."""
    return render_chat(tokenizer, example, evidence_for_condition(example, "code_input"))
