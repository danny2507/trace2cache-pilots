"""Prompt construction with condition-controlled runtime evidence."""

from __future__ import annotations

import json
from typing import Literal

from .benchmark import RepairCase
from .runtime import execute_with_trace, to_json_trace, to_text_trace


Condition = Literal[
    "test_only", "compact_text", "full_text", "structured_json", "shuffled_json"
]
CONDITIONS: tuple[Condition, ...] = (
    "test_only",
    "compact_text",
    "full_text",
    "structured_json",
    "shuffled_json",
)


def build_prompt(
    case: RepairCase,
    condition: Condition,
    *,
    evidence_case: RepairCase | None = None,
) -> tuple[str, dict[str, int | str]]:
    test = case.public_test
    trace = execute_with_trace(case.buggy_source, case.function_name, test.args)
    evidence_source = evidence_case or case
    evidence_test = evidence_source.public_test
    evidence_trace = execute_with_trace(
        evidence_source.buggy_source, evidence_source.function_name, evidence_test.args
    )
    actual = trace.error if trace.error else trace.output
    evidence = ""
    if condition == "compact_text":
        evidence = "\nCompact execution trace:\n" + to_text_trace(trace, compact=True)
    elif condition == "full_text":
        evidence = "\nFull execution trace:\n" + to_text_trace(trace)
    elif condition == "structured_json":
        evidence = "\nStructured execution trace (JSON):\n" + to_json_trace(trace)
    elif condition == "shuffled_json":
        if evidence_case is None:
            raise ValueError("shuffled_json requires an evidence_case")
        evidence = "\nStructured execution trace (JSON):\n" + to_json_trace(evidence_trace)
    elif condition != "test_only":
        raise ValueError(f"unknown condition: {condition}")

    args_text = ", ".join(repr(arg) for arg in test.args)
    prompt = f"""Fix the Python function below.

Specification: {case.description}

Buggy code:
```python
{case.buggy_source.rstrip()}
```

Failing test:
`{case.function_name}({args_text})` expected `{test.expected!r}` but produced `{actual}`.{evidence}

Return only the complete corrected function in one Python code block. Do not include tests or explanation.
"""
    measured_trace = evidence_trace if condition == "shuffled_json" else trace
    metadata: dict[str, int | str] = {
        "runtime_events": len(measured_trace.events),
        "trace_text_chars": len(to_text_trace(measured_trace)),
        "compact_trace_chars": len(to_text_trace(measured_trace, compact=True)),
        "json_trace_chars": len(to_json_trace(measured_trace)),
        "evidence_case_id": evidence_source.case_id,
    }
    return prompt, metadata
