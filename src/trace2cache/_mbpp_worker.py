"""Private subprocess worker for MBPP assertions."""

from __future__ import annotations

import json
import linecache
import sys

PROGRAM_FILENAME = "<mbpp-program>"


def _safe_value(value, max_chars: int = 120) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # noqa: BLE001
        rendered = f"<{type(value).__name__}: repr failed: {type(exc).__name__}>"
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3] + "..."
    return rendered


def _execute_test(test: str, namespace: dict, *, collect_trace: bool, max_events: int):
    events = []

    def tracer(frame, event, argument):
        if frame.f_code.co_filename != PROGRAM_FILENAME:
            return tracer
        if len(events) >= max_events:
            raise RuntimeError(f"trace exceeded max_events={max_events}")
        if event in {"call", "line", "return", "exception"}:
            value = None
            if event == "return":
                value = _safe_value(argument)
            elif event == "exception":
                exc_type, exc_value, _ = argument
                value = f"{exc_type.__name__}: {_safe_value(exc_value)}"
            events.append(
                {
                    "step": len(events),
                    "event": event,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                    "source": linecache.getline(PROGRAM_FILENAME, frame.f_lineno).strip(),
                    "locals": {
                        name: _safe_value(value)
                        for name, value in sorted(frame.f_locals.items())
                    },
                    "value": value,
                }
            )
        return tracer

    try:
        if collect_trace:
            sys.settrace(tracer)
        exec(compile(test, "<mbpp-test>", "exec"), namespace)  # noqa: S102
        outcome = "pass"
    except AssertionError:
        outcome = "fail"
    except Exception as exc:  # noqa: BLE001 -- runtime error is an outcome
        outcome = f"error:{type(exc).__name__}"
    finally:
        sys.settrace(None)
    return outcome, events


def main() -> None:
    payload = json.loads(sys.stdin.read())
    namespace = {"__name__": "__mbpp__"}
    source = payload["source"]
    source_lines = source.splitlines(keepends=True)
    linecache.cache[PROGRAM_FILENAME] = (
        len(source),
        None,
        source_lines,
        PROGRAM_FILENAME,
    )
    try:
        # Some MBPP setup snippets instantiate classes defined by the solution.
        exec(compile(source, PROGRAM_FILENAME, "exec"), namespace)  # noqa: S102
        exec(compile(payload.get("setup_source", ""), "<mbpp-setup>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 -- report benchmark runtime errors
        print(json.dumps({"status": "load_error", "tests": [], "error": type(exc).__name__}))
        return

    outcomes = []
    traces = []
    for test in payload["tests"]:
        outcome, events = _execute_test(
            test,
            namespace,
            collect_trace=payload.get("collect_trace", False),
            max_events=payload.get("max_events", 512),
        )
        outcomes.append(outcome)
        traces.append(events)
    if outcomes and all(outcome == "pass" for outcome in outcomes):
        status = "all_pass"
    elif "pass" in outcomes:
        status = "mixed"
    else:
        status = "all_fail"
    result = {"status": status, "tests": outcomes}
    if payload.get("collect_trace", False):
        result["traces"] = traces
    print(json.dumps(result))
    linecache.cache.pop(PROGRAM_FILENAME, None)


if __name__ == "__main__":
    main()
