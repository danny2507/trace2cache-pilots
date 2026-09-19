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


def permute_runtime_test_bindings(view: EvidenceView) -> EvidenceView:
    """Keep runtime facts but move same-role packets across executions.

    I/O rows are copied exactly.  Every intermediate row retains its own event id, test id,
    step and role, while the pair ``(content, source_line)`` is received from a different
    test whenever that role occurs in multiple tests.  Interleaving by within-test rank before
    a one-position rotation makes the permutation deterministic and avoids contiguous events
    from one test simply being exchanged among themselves.
    """
    events = list(view.events)
    by_role: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        if event.role_id not in IO_ROLES:
            by_role.setdefault(event.role_id, []).append(index)
    moved = 0
    for indices in by_role.values():
        by_test: dict[int, list[int]] = {}
        for index in indices:
            by_test.setdefault(events[index].test_id, []).append(index)
        if len(by_test) < 2:
            continue
        interleaved: list[int] = []
        for rank in range(max(len(group) for group in by_test.values())):
            for test_id in sorted(by_test):
                if rank < len(by_test[test_id]):
                    interleaved.append(by_test[test_id][rank])
        donors = interleaved[1:] + interleaved[:1]
        originals = list(events)
        for recipient, donor in zip(interleaved, donors):
            if originals[recipient].test_id != originals[donor].test_id:
                moved += 1
            events[recipient] = replace(
                originals[recipient],
                content=originals[donor].content,
                source_line=originals[donor].source_line,
            )
    if moved == 0:
        raise ValueError("runtime binding permutation found no cross-test runtime packets")
    return EvidenceView(view.view_uid + ":runtime_test_bindings_permuted", tuple(events))


def permute_runtime_temporal_bindings(view: EvidenceView) -> EvidenceView:
    """Break temporal/def-use binding without moving any runtime fact across tests.

    For each test execution, the complete multiset of intermediate ``(content, source_line)``
    packets is preserved exactly, but packets are cyclically reassigned to different runtime
    event rows. I/O rows and all row metadata (event id, test, step, role) remain unchanged.
    """
    events = list(view.events)
    by_test: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        if event.role_id not in IO_ROLES:
            by_test.setdefault(event.test_id, []).append(index)
    moved = 0
    for indices in by_test.values():
        if len(indices) < 2:
            continue
        originals = list(events)
        donors = indices[1:] + indices[:1]
        for recipient, donor in zip(indices, donors):
            moved += (originals[recipient].content, originals[recipient].source_line) != (
                originals[donor].content, originals[donor].source_line
            )
            events[recipient] = replace(
                originals[recipient], content=originals[donor].content,
                source_line=originals[donor].source_line,
            )
    if moved == 0:
        raise ValueError("temporal binding permutation found no distinct runtime packets")
    return EvidenceView(view.view_uid + ":runtime_temporal_bindings_permuted", tuple(events))


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
