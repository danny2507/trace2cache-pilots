"""Auditable structural relations over curated structured-runtime evidence.

These are intentionally conservative.  ``LAST_OBSERVED_VALUE`` is not asserted to be a
dynamic def-use edge: a trace callback exposes locals, not which name an expression read.
It means only that an earlier event was the last snapshot where that variable's tagged
value changed.  The encoder receives relation topology, never variable names.
"""

from __future__ import annotations

import json
from typing import Iterable


NONE, SELF, NEXT, SAME_SOURCE, LAST_OBSERVED_VALUE = range(5)
RELATION_NAMES = ("NONE", "SELF", "NEXT", "SAME_SOURCE", "LAST_OBSERVED_VALUE")
NUM_RELATIONS = len(RELATION_NAMES)


def _state(content: str) -> dict[str, object] | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    state = payload.get("state") if isinstance(payload, dict) else None
    return state if isinstance(state, dict) and all(isinstance(name, str) for name in state) else None


def relation_edges(events: Iterable[object]) -> tuple[tuple[int, int, int], ...]:
    """Return directed ``(source, target, relation_type)`` edges for one evidence view."""
    events = list(events)
    edges: list[tuple[int, int, int]] = []
    previous_by_test: dict[int, int] = {}
    previous_by_source: dict[tuple[int, int], int] = {}
    last_changed: dict[tuple[int, str], int] = {}
    previous_state: dict[int, dict[str, object]] = {}
    for index, event in enumerate(events):
        test_id = int(event.test_id)
        if test_id in previous_by_test: edges.append((previous_by_test[test_id], index, NEXT))
        previous_by_test[test_id] = index
        source_line = getattr(event, "source_line", None)
        if source_line is not None:
            source_key = (test_id, int(source_line))
            if source_key in previous_by_source: edges.append((previous_by_source[source_key], index, SAME_SOURCE))
            previous_by_source[source_key] = index
        current = _state(event.content)
        if current is None: continue
        old = previous_state.get(test_id, {})
        for name, value in current.items():
            key = (test_id, name)
            if key in last_changed: edges.append((last_changed[key], index, LAST_OBSERVED_VALUE))
            if name not in old or old[name] != value: last_changed[key] = index
        # Deleted locals must not point to an unrelated future rebinding.
        for name in set(old) - set(current): last_changed.pop((test_id, name), None)
        previous_state[test_id] = current
    return tuple(edges)
