"""Deterministic line-level tracing for trusted benchmark programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import linecache
import sys
from types import FrameType
from typing import Any


TRACE_FILENAME = "<trace2cache-benchmark>"


def safe_value(value: Any, max_chars: int = 120) -> str:
    """Return a bounded repr without recursively walking arbitrary objects."""
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive for hostile reprs
        rendered = f"<{type(value).__name__}: repr failed: {type(exc).__name__}>"
    if len(rendered) > max_chars:
        return rendered[: max_chars - 3] + "..."
    return rendered


@dataclass(frozen=True)
class TraceEvent:
    step: int
    event: str
    function: str
    line: int
    source: str
    locals: dict[str, str]
    value: str | None = None


@dataclass(frozen=True)
class TraceResult:
    events: list[TraceEvent]
    output: str | None
    error: str | None


def execute_with_trace(
    source: str,
    function_name: str,
    args: list[Any],
    *,
    max_events: int = 10_000,
) -> TraceResult:
    """Compile trusted source and trace one function call.

    This collector is for curated benchmark programs, not generated patches. Generated code is
    validated and run by :mod:`trace2cache.sandbox` in a subprocess.
    """
    source_lines = source.splitlines(keepends=True)
    linecache.cache[TRACE_FILENAME] = (len(source), None, source_lines, TRACE_FILENAME)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    compiled = compile(source, TRACE_FILENAME, "exec")
    exec(compiled, namespace)
    function = namespace[function_name]
    events: list[TraceEvent] = []

    def tracer(frame: FrameType, event: str, arg: Any):
        if frame.f_code.co_filename != TRACE_FILENAME:
            return None
        if len(events) >= max_events:
            raise RuntimeError(f"trace exceeded max_events={max_events}")
        if event in {"call", "line", "return", "exception"}:
            value = None
            if event == "return":
                value = safe_value(arg)
            elif event == "exception":
                exc_type, exc_value, _ = arg
                value = f"{exc_type.__name__}: {safe_value(exc_value)}"
            events.append(
                TraceEvent(
                    step=len(events),
                    event=event,
                    function=frame.f_code.co_name,
                    line=frame.f_lineno,
                    source=linecache.getline(TRACE_FILENAME, frame.f_lineno).strip(),
                    locals={k: safe_value(v) for k, v in sorted(frame.f_locals.items())},
                    value=value,
                )
            )
        return tracer

    output: str | None = None
    error: str | None = None
    previous = sys.gettrace()
    try:
        sys.settrace(tracer)
        output = safe_value(function(*args))
    except Exception as exc:
        error = f"{type(exc).__name__}: {safe_value(exc)}"
    finally:
        sys.settrace(previous)
        linecache.cache.pop(TRACE_FILENAME, None)
    return TraceResult(events=events, output=output, error=error)


def to_json_trace(result: TraceResult) -> str:
    return json.dumps(
        {
            "events": [asdict(event) for event in result.events],
            "output": result.output,
            "error": result.error,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def to_text_trace(result: TraceResult, *, compact: bool = False) -> str:
    events = result.events
    if compact:
        events = _changed_state_events(events)
    lines = []
    for event in events:
        state = ", ".join(f"{name}={value}" for name, value in event.locals.items())
        suffix = f" -> {event.value}" if event.value is not None else ""
        lines.append(
            f"#{event.step} {event.event} L{event.line} `{event.source}` [{state}]{suffix}"
        )
    if result.error:
        lines.append(f"ERROR {result.error}")
    else:
        lines.append(f"OUTPUT {result.output}")
    return "\n".join(lines)


def _changed_state_events(events: list[TraceEvent]) -> list[TraceEvent]:
    """Keep boundary events and line events whose state differs from the previous event."""
    if not events:
        return []
    kept: list[TraceEvent] = []
    previous_locals: dict[str, str] | None = None
    for event in events:
        boundary = event.event in {"call", "return", "exception"}
        changed = previous_locals is None or event.locals != previous_locals
        if boundary or changed:
            kept.append(event)
        previous_locals = event.locals
    return kept

