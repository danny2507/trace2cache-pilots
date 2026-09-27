"""Private subprocess worker for MBPP assertions."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import linecache
import sys

PROGRAM_FILENAME = "<mbpp-program>"
STDIO_PREFIX = "# TRACE2CACHE_STDIO "
OBSERVED_CHARS = 240


def _safe_value(value, max_chars: int = 120) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # noqa: BLE001
        rendered = f"<{type(value).__name__}: repr failed: {type(exc).__name__}>"
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3] + "..."
    return rendered


def _parse_stdio_test(test: str):
    if not str(test).startswith(STDIO_PREFIX):
        return None
    payload = json.loads(test[len(STDIO_PREFIX) :])
    return str(payload["stdin"]), str(payload["stdout"])


def _clip_observed(value, max_chars: int = OBSERVED_CHARS) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else _safe_value(value, max_chars)
    collapsed = "\\n".join(text.replace("\r\n", "\n").split("\n"))
    if len(collapsed) > max_chars:
        return collapsed[: max_chars - 3] + "..."
    return collapsed


def _assert_comparison(test: str):
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    return ast.unparse(expr.left), ast.unparse(expr.comparators[0])


def _eval_observed(source: str, namespace: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _clip_observed(eval(compile(source, "<mbpp-observed>", "eval"), namespace))  # noqa: S307


def _last_trace_value(events: list):
    for event in reversed(events):
        if event.get("event") in {"return", "exception"} and event.get("value"):
            return _clip_observed(str(event["value"]))
    return ""


def _empty_observed(expected: str = "") -> dict:
    return {"got": "", "expected": _clip_observed(expected), "error": None}


def _tracer(events: list, max_events: int):
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

    return tracer


def _execute_stdio_program(
    source: str, stdin_text: str, expected: str, *, collect_trace: bool, max_events: int
):
    events = []
    namespace = {"__name__": "__main__"}
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        if collect_trace:
            sys.settrace(_tracer(events, max_events))
        sys.stdin = io.StringIO(stdin_text)
        sys.stdout = buf_out
        sys.stderr = buf_err
        try:
            exec(compile(source, PROGRAM_FILENAME, "exec"), namespace)  # noqa: S102
        except SystemExit:
            pass
        got = buf_out.getvalue().replace("\r\n", "\n")
        exp = expected.replace("\r\n", "\n")
        outcome = "pass" if got.rstrip("\n") == exp.rstrip("\n") else "fail"
        observed = {
            "got": _clip_observed(got.rstrip("\n")),
            "expected": _clip_observed(exp.rstrip("\n")),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        outcome = f"error:{type(exc).__name__}"
        observed = {
            "got": _clip_observed(buf_out.getvalue().rstrip("\n")),
            "expected": _clip_observed(expected.rstrip("\n")),
            "error": type(exc).__name__,
        }
    finally:
        sys.settrace(None)
        sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    return outcome, events, observed


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

    observed = _empty_observed()
    parts = _assert_comparison(test)
    if parts is not None:
        try:
            observed["expected"] = _eval_observed(parts[1], namespace)
        except Exception:  # noqa: BLE001
            observed["expected"] = _clip_observed(parts[1])
    try:
        if collect_trace:
            sys.settrace(tracer)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(test, "<mbpp-test>", "exec"), namespace)  # noqa: S102
        outcome = "pass"
        if parts is not None and not observed["got"]:
            observed["got"] = observed["expected"]
    except AssertionError:
        outcome = "fail"
        if parts is not None:
            try:
                observed["got"] = _eval_observed(parts[0], namespace)
            except Exception as exc:  # noqa: BLE001
                observed["error"] = type(exc).__name__
        if not observed["got"]:
            observed["got"] = _last_trace_value(events)
    except Exception as exc:  # noqa: BLE001
        outcome = f"error:{type(exc).__name__}"
        observed["error"] = type(exc).__name__
        observed["got"] = _last_trace_value(events) or _clip_observed(exc)
    finally:
        sys.settrace(None)
    return outcome, events, observed


def _execute_call(call: str, namespace: dict, *, collect_trace: bool, max_events: int):
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
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            value = eval(compile(call, "<mbpp-call>", "eval"), namespace)
        result = {"status": "ok", "output": _safe_value(value), "trace": events}
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": f"error:{type(exc).__name__}",
            "output": _safe_value(exc),
            "trace": events,
        }
    finally:
        sys.settrace(None)
    return result


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
    tests = payload.get("tests") or []
    parsed_stdio = [_parse_stdio_test(test) for test in tests]
    if tests and all(item is not None for item in parsed_stdio):
        outcomes = []
        traces = []
        observed_rows = []
        for stdin_text, expected in parsed_stdio:
            outcome, events, observed = _execute_stdio_program(
                source,
                stdin_text,
                expected,
                collect_trace=payload.get("collect_trace", False),
                max_events=payload.get("max_events", 512),
            )
            outcomes.append(outcome)
            traces.append(events)
            observed_rows.append(observed)
        if outcomes and all(outcome == "pass" for outcome in outcomes):
            status = "all_pass"
        elif "pass" in outcomes:
            status = "mixed"
        else:
            status = "all_fail"
        result = {"status": status, "tests": outcomes, "observed": observed_rows}
        if payload.get("collect_trace", False):
            result["traces"] = traces
        print(json.dumps(result))
        linecache.cache.pop(PROGRAM_FILENAME, None)
        return
    try:
        # Some MBPP setup snippets instantiate classes defined by the solution.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(source, PROGRAM_FILENAME, "exec"), namespace)  # noqa: S102
            exec(  # noqa: S102
                compile(payload.get("setup_source", ""), "<mbpp-setup>", "exec"), namespace
            )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "load_error", "tests": [], "error": type(exc).__name__}))
        return

    if "calls" in payload:
        calls = [
            _execute_call(
                call,
                namespace,
                collect_trace=payload.get("collect_trace", False),
                max_events=payload.get("max_events", 512),
            )
            for call in payload["calls"]
        ]
        print(json.dumps({"status": "ok", "calls": calls}))
        linecache.cache.pop(PROGRAM_FILENAME, None)
        return

    outcomes = []
    traces = []
    observed_rows = []
    for test in payload["tests"]:
        outcome, events, observed = _execute_test(
            test,
            namespace,
            collect_trace=payload.get("collect_trace", False),
            max_events=payload.get("max_events", 512),
        )
        outcomes.append(outcome)
        traces.append(events)
        observed_rows.append(observed)
    if outcomes and all(outcome == "pass" for outcome in outcomes):
        status = "all_pass"
    elif "pass" in outcomes:
        status = "mixed"
    else:
        status = "all_fail"
    result = {"status": status, "tests": outcomes, "observed": observed_rows}
    if payload.get("collect_trace", False):
        result["traces"] = traces
    print(json.dumps(result))
    linecache.cache.pop(PROGRAM_FILENAME, None)


if __name__ == "__main__":
    main()
