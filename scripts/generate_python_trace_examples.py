#!/usr/bin/env python3
"""Generate executable Python bug/trace examples without an external LLM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from trace2cache.benchmark import get_cases
from trace2cache.runtime import TraceEvent, execute_with_trace, to_text_trace


def event_role(event: TraceEvent, changed: dict[str, dict[str, str | None]]) -> str:
    stripped = event.source.strip()
    if event.event == "call":
        return "call_input"
    if event.event == "return":
        return "return_sink"
    if event.event == "exception":
        return "exception_sink"
    if stripped.startswith("if ") or stripped.startswith("elif "):
        return "branch_predicate"
    if stripped.startswith("for ") or stripped.startswith("while "):
        return "loop_control"
    if changed:
        return "state_definition"
    return "executed_line"


def annotate_events(events: list[TraceEvent]) -> list[dict[str, object]]:
    annotated = []
    previous: dict[str, str] = {}
    for event in events:
        names = sorted(set(previous) | set(event.locals))
        changed = {
            name: {"before": previous.get(name), "after": event.locals.get(name)}
            for name in names
            if previous.get(name) != event.locals.get(name)
        }
        payload = asdict(event)
        payload["changed"] = changed
        payload["semantic_role"] = event_role(event, changed)
        payload["on_compact_slice"] = bool(
            changed
            or event.event in {"call", "return", "exception"}
            or payload["semantic_role"] in {"branch_predicate", "loop_control"}
        )
        annotated.append(payload)
        previous = event.locals
    return annotated


def make_assertion(function_name: str, args: list[object], expected: object) -> str:
    rendered_args = ", ".join(repr(argument) for argument in args)
    return f"assert {function_name}({rendered_args}) == {expected!r}"


def build_examples(limit: int | None = None) -> list[dict[str, object]]:
    examples = []
    for case in get_cases(limit):
        result = execute_with_trace(
            case.buggy_source,
            case.function_name,
            case.public_test.args,
        )
        expected_repr = repr(case.public_test.expected)
        examples.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "language": "python",
                "function_name": case.function_name,
                "buggy_source": case.buggy_source,
                "failing_test": make_assertion(
                    case.function_name, case.public_test.args, case.public_test.expected
                ),
                "test_args": case.public_test.args,
                "expected": expected_repr,
                "actual": result.output,
                "error": result.error,
                "test_passed": result.error is None and result.output == expected_repr,
                "trace_event_count": len(result.events),
                "raw_text_trace": to_text_trace(result),
                "compact_text_trace": to_text_trace(result, compact=True),
                "structured_events": annotate_events(result.events),
            }
        )
    return examples


def markdown_report(examples: list[dict[str, object]]) -> str:
    sections = ["# Executed Python trace examples", ""]
    for example in examples:
        sections.extend(
            [
                f"## {example['case_id']}",
                "",
                str(example["description"]),
                "",
                "```python",
                str(example["buggy_source"]).rstrip(),
                "",
                str(example["failing_test"]),
                "```",
                "",
                f"Expected: `{example['expected']}`; actual: `{example['actual']}`; "
                f"events: {example['trace_event_count']}.",
                "",
                "```text",
                str(example["compact_text_trace"]),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--jsonl", default="artifacts/real_traces/python_bug_traces.jsonl"
    )
    parser.add_argument(
        "--markdown", default="artifacts/real_traces/python_bug_traces.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(args.limit)
    jsonl = Path(args.jsonl)
    markdown = Path(args.markdown)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples)
    )
    markdown.write_text(markdown_report(examples))
    print(
        json.dumps(
            {
                "examples": len(examples),
                "failed_tests": sum(not example["test_passed"] for example in examples),
                "total_events": sum(int(example["trace_event_count"]) for example in examples),
                "jsonl": str(jsonl),
                "markdown": str(markdown),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
