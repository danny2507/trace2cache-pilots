"""Deterministic event selection for model-native program-repair pilots."""

from __future__ import annotations

from .runtime import TraceEvent


def changed_locals(
    previous: dict[str, str], current: dict[str, str]
) -> dict[str, tuple[str | None, str | None]]:
    return {
        name: (previous.get(name), current.get(name))
        for name in sorted(set(previous) | set(current))
        if previous.get(name) != current.get(name)
    }


def summarize_events(events: list[TraceEvent]) -> list[str]:
    """Render state deltas rather than repeatedly serializing the full frame."""
    summaries: list[str] = []
    previous: dict[str, str] = {}
    for event in events:
        changes = changed_locals(previous, event.locals)
        boundary = event.event in {"call", "return", "exception"}
        control = event.source.strip().startswith(("if ", "elif ", "for ", "while "))
        if changes or boundary or control:
            fields = [f"{name}: {before} -> {after}" for name, (before, after) in changes.items()]
            state = "; ".join(fields) if fields else "no state change"
            value = f"; outcome={event.value}" if event.value is not None else ""
            summaries.append(
                f"{event.event} line {event.line} `{event.source}`; {state}{value}"
            )
        previous = event.locals
    return summaries


def select_temporal_events(summaries: list[str], limit: int) -> list[str]:
    """Keep endpoints and uniformly cover the intervening execution."""
    if limit < 1:
        raise ValueError("event limit must be positive")
    if len(summaries) <= limit:
        return summaries
    if limit == 1:
        return [summaries[-1]]
    indices = []
    for position in range(limit):
        index = round(position * (len(summaries) - 1) / (limit - 1))
        if index not in indices:
            indices.append(index)
    # Rounding can theoretically collapse indices; fill deterministically.
    for index in range(len(summaries)):
        if len(indices) == limit:
            break
        if index not in indices:
            indices.append(index)
    return [summaries[index] for index in sorted(indices)]
