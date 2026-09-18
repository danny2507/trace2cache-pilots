"""Predeclared Stage-C evidence views derived from one audited runtime record."""

from __future__ import annotations

import json
from dataclasses import replace

from .paired_evidence import ACTUAL, EXPECTED, INPUT, STATUS, TEST_START, EvidenceEvent, EvidenceView


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
    """Stable textual serialization with no facts beyond the supplied event view."""
    lines = []
    for event in view.events:
        lines.append(
            json.dumps(
                {
                    "event_id": event.event_id,
                    "test_id": event.test_id,
                    "step": event.step,
                    "role_id": event.role_id,
                    "source_line": event.source_line,
                    "content": json.loads(event.content),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines)
