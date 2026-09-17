"""Typed, single-execution trace collection for the paired-runtime v2 pilot.

This intentionally does not replace :mod:`trace2cache.runtime`: historical pilots use
its bounded ``repr`` protocol.  The collector here is only for trusted, curated sources.
It snapshots values from each trace callback, so a later in-place mutation cannot rewrite
an earlier state observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import linecache
import math
import sys
from types import FrameType
from typing import Any


TRACE_FILENAME = "<trace2cache-structured-runtime>"
_MAX_DEPTH = 4
_MAX_ITEMS = 64
_MAX_CHARS = 256


@dataclass(frozen=True)
class StructuredTraceEvent:
    step: int
    event: str
    function: str
    line: int
    source: str
    # Tagged, JSON-compatible values captured before the next Python line executes.
    locals: dict[str, dict[str, Any]]
    value: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredTraceResult:
    events: tuple[StructuredTraceEvent, ...]
    output: dict[str, Any] | None
    error: dict[str, Any] | None
    truncated: bool = False


def tagged_value(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """Losslessly tag the small value language used by curated pilot programs.

    Unknown objects and oversized values are explicit rather than silently converted to a
    potentially ambiguous repr.  Bool is checked before int because Python subclasses it.
    """
    if depth >= _MAX_DEPTH:
        return {"type": "unknown", "reason": "max_depth"}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"type": "float", "value": value}
        return {"type": "float", "value": repr(value)}
    if isinstance(value, str):
        if len(value) > _MAX_CHARS:
            return {"type": "str", "value": value[:_MAX_CHARS], "truncated": True}
        return {"type": "str", "value": value}
    if isinstance(value, (list, tuple)):
        items = [tagged_value(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
        result: dict[str, Any] = {"type": "list" if isinstance(value, list) else "tuple", "items": items}
        if len(value) > _MAX_ITEMS:
            result["truncated"] = True
        return result
    return {"type": "unknown", "reason": type(value).__name__}


def canonical_value(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def values_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    """Type-aware equality for tagged values (not Python's ``True == 1`` equality)."""
    return left == right


def execute_structured_trace(
    source: str,
    function_name: str,
    args: list[Any],
    *,
    max_events: int = 512,
) -> StructuredTraceResult:
    """Compile trusted source, trace one call, and capture the typed return in that call."""
    source_lines = source.splitlines(keepends=True)
    linecache.cache[TRACE_FILENAME] = (len(source), None, source_lines, TRACE_FILENAME)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    compiled = compile(source, TRACE_FILENAME, "exec")
    exec(compiled, namespace)
    function = namespace[function_name]
    events: list[StructuredTraceEvent] = []
    collector_truncated = False

    def tracer(frame: FrameType, event: str, arg: Any):
        nonlocal collector_truncated
        if frame.f_code.co_filename != TRACE_FILENAME:
            return None
        if event not in {"call", "line", "return", "exception"}:
            return tracer
        if len(events) >= max_events:
            collector_truncated = True
            return tracer
        event_value: dict[str, Any] | None = None
        if event == "return":
            event_value = tagged_value(arg)
        elif event == "exception":
            exc_type, exc_value, _ = arg
            event_value = {"type": "exception", "name": exc_type.__name__, "message": str(exc_value)[:_MAX_CHARS]}
        # Serialize now; storing frame.f_locals would observe future list mutations.
        snapshot = {name: tagged_value(value) for name, value in sorted(frame.f_locals.items())}
        events.append(
            StructuredTraceEvent(
                step=len(events), event=event, function=frame.f_code.co_name,
                line=frame.f_lineno,
                source=linecache.getline(TRACE_FILENAME, frame.f_lineno).strip(),
                locals=snapshot, value=event_value,
            )
        )
        return tracer

    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    previous = sys.gettrace()
    try:
        sys.settrace(tracer)
        output = tagged_value(function(*args))
    except Exception as exc:  # Curated task failures are evidence, not an unhandled crash.
        error = {"type": "exception", "name": type(exc).__name__, "message": str(exc)[:_MAX_CHARS]}
    finally:
        sys.settrace(previous)
        linecache.cache.pop(TRACE_FILENAME, None)
    return StructuredTraceResult(tuple(events), output, error, collector_truncated)
