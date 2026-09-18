"""Predeclared Stage-C evidence views derived from one audited runtime record."""

from __future__ import annotations

import json
from dataclasses import replace

from .paired_evidence import (
    ACTUAL,
    EXPECTED,
    INPUT,
    STATUS,
    TEST_START,
    ROLE_NAMES,
    EvidenceEvent,
    EvidenceView,
)


IO_ROLES = frozenset((TEST_START, INPUT, ACTUAL, EXPECTED, STATUS))


def io_only(view: EvidenceView) -> EvidenceView:
    """Keep only test identity, inputs, observed/expected output, and derived status."""
    events = tuple(event for event in view.events if event.role_id in IO_ROLES)
    if not events:
        raise ValueError("I/O control unexpectedly removed every event")
    return EvidenceView(view.view_uid + ":io_only", events)


def corrupt_runtime_keep_io(view: EvidenceView) -> EvidenceView:
    """Retain exact I/O but erase intermediate execution payloads and source locations."""
    events = []
    for event in view.events:
        if event.role_id in IO_ROLES:
            events.append(event)
        else:
            payload = json.dumps({"runtime": "CORRUPTED"}, sort_keys=True, separators=(",", ":"))
            events.append(replace(event, content=payload, source_line=None))
    return EvidenceView(view.view_uid + ":runtime_corrupted_keep_io", tuple(events))


def structured_text(view: EvidenceView) -> str:
    """Lossless, role-named text serialization with no facts beyond ``view``."""
    lines = []
    for event in view.events:
        lines.append(
            json.dumps(
                {
                    "event_id": event.event_id,
                    "test_id": event.test_id,
                    "step": event.step,
                    "role": ROLE_NAMES[event.role_id],
                    "source_line": event.source_line,
                    "content": json.loads(event.content),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines)


def compact_behavioral_text(view: EvidenceView) -> str:
    """Readable trace text grouped by test, retaining every supplied event payload.

    This intentionally changes only the receiver-facing interface: it neither derives a
    new fact nor removes a runtime event.  The short labels make role semantics available
    to a frozen language model without requiring it to infer a numeric schema.
    """
    lines: list[str] = []
    active_test: int | None = None
    for event in view.events:
        if event.test_id != active_test:
            if active_test is not None:
                lines.append("END_TEST")
            active_test = event.test_id
            lines.append(f"TEST {active_test}")
        location = "" if event.source_line is None else f" line={event.source_line}"
        payload = json.loads(event.content)
        lines.append(
            f"{ROLE_NAMES[event.role_id]} step={event.step}{location}: "
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
    if active_test is not None:
        lines.append("END_TEST")
    return "\n".join(lines)
