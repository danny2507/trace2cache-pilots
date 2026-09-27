"""Frozen MBPP repair cohort with disjoint public evidence and hidden tests.

The evaluation question is whether eight model-native latent states beat a matched
textual trace on held-out assertions. Canonical source may be used as an offline
oracle; it never belongs in a test-split prompt or adapter update.
"""

from __future__ import annotations

import ast
import difflib
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .mbpp import MBPPMutation, MBPPTask, generate_mutants, run_mbpp_tests, trace_mbpp_tests

SCHEMA_VERSION = 1
MARKER = "<TRACE2CACHE_REPAIR_CODE>"
STDIO_PREFIX = "# TRACE2CACHE_STDIO "
TEST, CALL, BRANCH, STATE, LINE, RETURN, PASS, FAIL = range(8)
ROLE_NAMES = ("TEST", "CALL", "BRANCH", "STATE", "LINE", "RETURN", "PASS", "FAIL")
IO_ROLES = frozenset((TEST, PASS, FAIL))
RUNTIME_ROLES = frozenset((CALL, BRANCH, STATE, LINE, RETURN))
SAFE_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "io",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "string",
    "sys",
}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}


@dataclass(frozen=True)
class EvidenceEvent:
    test_id: int
    role_id: int
    content: str


@dataclass(frozen=True)
class Partition:
    public_tests: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    public_hashes: tuple[str, ...]
    hidden_hashes: tuple[str, ...]
    reason: str = "ok"


@dataclass(frozen=True)
class RepairExample:
    example_id: str
    task_id: int
    split: str
    mutation_kind: str
    mutation_ordinal: int
    description: str
    buggy_source: str
    target_source: str
    setup_source: str
    failing_public_test: str
    public_tests: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    public_hashes: tuple[str, ...]
    hidden_hashes: tuple[str, ...]
    events: tuple[EvidenceEvent, ...]
    source_hash: str
    mutant_public_status: str
    mutant_hidden_status: str


def encode_stdio_test(stdin: str, stdout: str) -> str:
    return STDIO_PREFIX + json.dumps({"stdin": stdin, "stdout": stdout}, ensure_ascii=False)


def parse_stdio_test(test: str) -> tuple[str, str] | None:
    if not str(test).startswith(STDIO_PREFIX):
        return None
    payload = json.loads(test[len(STDIO_PREFIX) :])
    return str(payload["stdin"]), str(payload["stdout"])


def format_public_test(test: str) -> str:
    parsed = parse_stdio_test(test)
    if parsed is None:
        return f"`{test}`"
    stdin, stdout = parsed
    return (
        "Input:\n```\n"
        + stdin.rstrip("\n")
        + "\n```\nExpected output:\n```\n"
        + stdout.rstrip("\n")
        + "\n```"
    )


def test_event_text(assertion: str) -> str:
    parsed = parse_stdio_test(assertion)
    if parsed is None:
        return f"test assertion {assertion}"
    stdin, stdout = parsed
    return f"test stdin={stdin!r} expected={stdout!r}"


def outcome_event_text(outcome: str, assertion: str, observed: dict | None) -> str:
    """Haque error-prompt analogue: outcome plus expected vs got."""
    observed = observed or {}
    parsed = parse_stdio_test(assertion)
    expected = observed.get("expected")
    got = observed.get("got")
    error = observed.get("error")
    if parsed is not None:
        if expected in (None, ""):
            expected = parsed[1]
        if got in (None, "") and outcome == "pass":
            got = parsed[1]
    extra: list[str] = []
    if expected not in (None, ""):
        extra.append(f"expected {_clip_gist_line(str(expected), 120)}")
    if got not in (None, ""):
        extra.append(f"got {_clip_gist_line(str(got), 120)}")
    if error:
        extra.append(f"error {error}")
    if extra:
        return f"test outcome {outcome}: " + "; ".join(extra)
    return f"test outcome {outcome}"


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def call_key(function_name: str, arguments: list[object]) -> str:
    return f"{function_name}({', '.join(map(repr, arguments))})"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def entry_functions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def literal_calls(source: str, statements: Iterable[str]) -> list[str]:
    definitions = entry_functions(source)
    calls: list[str] = []
    seen: set[str] = set()
    for statement in statements:
        found = False
        try:
            tree = ast.parse(statement)
        except SyntaxError:
            raw = "raw:" + hash_key(statement.strip())
            if raw not in seen:
                seen.add(raw)
                calls.append(raw)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in definitions or node.keywords:
                continue
            try:
                arguments = [ast.literal_eval(argument) for argument in node.args]
            except (ValueError, TypeError):
                continue
            key = call_key(node.func.id, arguments)
            found = True
            if key not in seen:
                seen.add(key)
                calls.append(key)
        if not found:
            raw = "raw:" + hash_key(statement.strip())
            if raw not in seen:
                seen.add(raw)
                calls.append(raw)
    return calls


def select_events(events: list[dict], limit: int) -> list[dict]:
    selected = []
    previous_locals = None
    for event in events:
        boundary = event["event"] in {"call", "return", "exception"}
        changed = event["locals"] != previous_locals
        if boundary or changed:
            selected.append(event)
        previous_locals = event["locals"]
    if len(selected) <= limit:
        return selected
    indices = [round(index * (len(selected) - 1) / (limit - 1)) for index in range(limit)]
    return [selected[index] for index in indices]


def event_role(event: dict) -> int:
    if event["event"] == "call":
        return CALL
    if event["event"] in {"return", "exception"}:
        return RETURN
    source = event["source"].lstrip()
    if source.startswith(("if ", "elif ", "for ", "while ")):
        return BRANCH
    return STATE if event["locals"] else LINE


def event_text(event: dict) -> str:
    state = ", ".join(f"{name}={value}" for name, value in event["locals"].items())
    result = f" return={event['value']}" if event["value"] is not None else ""
    text = (
        f"{event['event']} {event['function']} line {event['line']} "
        f"source {event['source']} state {state}{result}"
    )
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)


def load_mbppplus_assertions(path: str | Path) -> dict[int, tuple[str, ...]]:
    mapping: dict[int, tuple[str, ...]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(str(row["task_id"]).split("/")[-1])
            assertions = tuple(
                statement.strip()
                for statement in str(row.get("assertion", "")).splitlines()
                if statement.strip().startswith("assert ")
            )
            if assertions:
                mapping[task_id] = assertions
    return mapping


def _status(outcomes: list[str]) -> str:
    if not outcomes:
        return "empty"
    if all(outcome == "pass" for outcome in outcomes):
        return "all_pass"
    if "pass" in outcomes:
        return "mixed"
    return "all_fail"


def partition_tests(
    tests: tuple[str, ...],
    canonical_outcomes: list[str],
    mutant_outcomes: list[str],
    *,
    source: str,
) -> Partition | None:
    if len(tests) != len(canonical_outcomes) or len(tests) != len(mutant_outcomes):
        return None
    if any(outcome != "pass" for outcome in canonical_outcomes):
        return None
    failing = [index for index, outcome in enumerate(mutant_outcomes) if outcome != "pass"]
    if not failing:
        return None
    public_index = failing[0]
    hidden_indices = [index for index in range(len(tests)) if index != public_index]
    if not hidden_indices:
        return None
    if all(mutant_outcomes[index] == "pass" for index in hidden_indices):
        return None
    public_tests = (tests[public_index],)
    hidden_tests = tuple(tests[index] for index in hidden_indices)
    public_hashes = tuple(hash_key(key) for key in literal_calls(source, public_tests))
    hidden_hashes = tuple(hash_key(key) for key in literal_calls(source, hidden_tests))
    if set(public_hashes) & set(hidden_hashes):
        return None
    return Partition(public_tests, hidden_tests, public_hashes, hidden_hashes)


def extra_hidden_tests(
    task: MBPPTask,
    partition: Partition,
    plus_assertions: tuple[str, ...],
    *,
    timeout_seconds: float = 2.0,
) -> tuple[str, ...]:
    candidates = []
    seen = set(partition.hidden_tests)
    public_hashes = set(partition.public_hashes)
    hidden_hashes = set(partition.hidden_hashes)
    for assertion in (*task.challenge_tests, *plus_assertions):
        if assertion in seen:
            continue
        keys = literal_calls(task.canonical_source, (assertion,))
        hashes = {hash_key(key) for key in keys}
        if hashes & public_hashes:
            continue
        candidates.append(assertion)
        seen.add(assertion)
        hidden_hashes.update(hashes)
    if not candidates:
        return ()
    probe = replace(task, tests=tuple(candidates))
    result = run_mbpp_tests(probe, timeout_seconds=timeout_seconds)
    if len(result.get("tests", [])) != len(candidates):
        return ()
    return tuple(
        assertion
        for assertion, outcome in zip(candidates, result["tests"])
        if outcome == "pass"
    )


def build_events(
    task: MBPPTask,
    mutation: MBPPMutation,
    public_tests: tuple[str, ...],
    *,
    max_events_per_test: int,
    timeout_seconds: float = 3.0,
) -> tuple[EvidenceEvent, ...] | None:
    probe = replace(task, tests=public_tests)
    traced = trace_mbpp_tests(
        probe, mutation.source, timeout_seconds=timeout_seconds, max_events=512
    )
    if traced.get("status") in {"timeout", "worker_error", "load_error"}:
        return None
    outcomes = traced.get("tests") or []
    traces = traced.get("traces") or [[] for _ in outcomes]
    observed_rows = traced.get("observed") or [{} for _ in outcomes]
    if len(outcomes) != len(public_tests) or len(traces) != len(public_tests):
        return None
    if len(observed_rows) != len(public_tests):
        observed_rows = list(observed_rows) + [{}] * (len(public_tests) - len(observed_rows))
    events: list[EvidenceEvent] = []
    for test_id, (assertion, outcome, trace, observed) in enumerate(
        zip(public_tests, outcomes, traces, observed_rows)
    ):
        events.append(EvidenceEvent(test_id, TEST, test_event_text(assertion)))
        for event in select_events(trace, max_events_per_test):
            events.append(EvidenceEvent(test_id, event_role(event), event_text(event)))
        events.append(
            EvidenceEvent(
                test_id,
                PASS if outcome == "pass" else FAIL,
                outcome_event_text(outcome, assertion, observed if isinstance(observed, dict) else None),
            )
        )
    return tuple(events)


def io_only_events(events: tuple[EvidenceEvent, ...]) -> tuple[EvidenceEvent, ...]:
    selected = tuple(event for event in events if event.role_id in IO_ROLES)
    if not selected:
        raise ValueError("I/O view removed every event")
    return selected


def corrupt_runtime_events(events: tuple[EvidenceEvent, ...]) -> tuple[EvidenceEvent, ...]:
    return tuple(
        event
        if event.role_id in IO_ROLES
        else EvidenceEvent(event.test_id, event.role_id, "CORRUPTED_RUNTIME")
        for event in events
    )


def compact_trace_text(events: tuple[EvidenceEvent, ...]) -> str:
    lines: list[str] = []
    active: int | None = None
    for event in events:
        if event.test_id != active:
            if active is not None:
                lines.append("END_TEST")
            active = event.test_id
            lines.append(f"TEST {event.test_id}")
        lines.append(f"{ROLE_NAMES[event.role_id]}: {event.content}")
    if active is not None:
        lines.append("END_TEST")
    return "\n".join(lines)


def _clip_gist_line(text: str, width: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 3] + "..."


def gist_events(events: tuple[EvidenceEvent, ...]) -> tuple[EvidenceEvent, ...]:
    """Keep I/O plus the last LINE, STATE, and RETURN. Drop loop-unrolled intermediates."""
    if not events:
        return events
    keep: set[int] = set()
    last_index: dict[int, int] = {}
    for index, event in enumerate(events):
        if event.role_id in IO_ROLES:
            keep.add(index)
        elif event.role_id in (LINE, STATE, RETURN):
            last_index[event.role_id] = index
    keep.update(last_index.values())
    if not keep:
        return events
    return tuple(events[index] for index in sorted(keep))


_RETURN_VALUE = re.compile(r"\breturn=(.*)$")
_EXPECTED_LITERAL = re.compile(r"\bexpected=(.*)$")
_GOT_FIELD = re.compile(r"\bgot ([^;]+)")
_EXPECTED_FIELD = re.compile(r"\bexpected ([^;]+)")
_ERROR_TAIL = re.compile(r"; error (.+)$")
_ERROR_TYPE = re.compile(r"outcome error:([^:]*)")


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if match is None:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return str(ast.literal_eval(value))
        except (SyntaxError, ValueError):
            return value
    return value


def discrepancy_fields(events: tuple[EvidenceEvent, ...]) -> tuple[str, str]:
    """Recover expected vs got from FAIL, TEST, or last RETURN."""
    expected = ""
    got = ""
    for event in events:
        if event.role_id == FAIL:
            expected = expected or _first_match(_EXPECTED_FIELD, event.content)
            got = (
                got
                or _first_match(_GOT_FIELD, event.content)
                or _first_match(_ERROR_TAIL, event.content)
                or _first_match(_ERROR_TYPE, event.content)
            )
        elif event.role_id == TEST:
            expected = expected or _first_match(_EXPECTED_LITERAL, event.content)
            if not expected and "==" in event.content:
                expected = event.content.rsplit("==", 1)[-1].strip().rstrip(")")
    if not got:
        for event in reversed(events):
            if event.role_id == RETURN:
                got = _first_match(_RETURN_VALUE, event.content)
                if got:
                    break
    return expected, got


def discrepancy_text(
    events: tuple[EvidenceEvent, ...],
    *,
    width: int = 180,
    swap: bool = False,
) -> str:
    """One-line expected vs got. No last LINE/STATE/RETURN and no gold REPLACE."""
    expected, got = discrepancy_fields(events)
    if swap:
        expected, got = got, expected
    if not expected and not got:
        return "unavailable"
    return _clip_gist_line(
        f"DISCREPANCY: expected {expected or '?'} got {got or '?'}",
        width,
    )


def gist_text(events: tuple[EvidenceEvent, ...], *, width: int = 180) -> str:
    """Short discrepancy note: I/O expected vs got, then last LINE/STATE/RETURN."""
    chosen = gist_events(events)
    if not chosen:
        return "unavailable"
    expected, got = discrepancy_fields(events)
    io_events = [event for event in chosen if event.role_id in IO_ROLES]
    runtime = [event for event in chosen if event.role_id not in IO_ROLES]
    lines: list[str] = []
    if expected or got:
        lines.append(discrepancy_text(events, width=width))
    lines.extend(
        f"{ROLE_NAMES[event.role_id]}: {_clip_gist_line(event.content, width)}" for event in io_events
    )
    if runtime:
        lines.append("LAST RUNTIME:")
        lines.extend(
            f"{ROLE_NAMES[event.role_id]}: {_clip_gist_line(event.content, width)}" for event in runtime
        )
    return "\n".join(lines) or "unavailable"


_LINE_NUMBER = re.compile(r"\bline (\d+)\b")
_SOURCE_FIELD = re.compile(r"\bsource (.*?)(?:\s+state\b|\s+return=|$)")


def failing_span(
    events: tuple[EvidenceEvent, ...],
    buggy_source: str = "",
) -> tuple[int | None, str]:
    """Last executed source line: exception if any, else last runtime event with source."""
    exception_event: EvidenceEvent | None = None
    last_runtime: EvidenceEvent | None = None
    for event in events:
        if event.role_id in IO_ROLES:
            continue
        line_no, source = _span_from_event(event, buggy_source)
        if not source:
            continue
        last_runtime = event
        if event.content.startswith("exception"):
            exception_event = event
    chosen = exception_event or last_runtime
    if chosen is None:
        return None, ""
    return _span_from_event(chosen, buggy_source)


def _span_from_event(event: EvidenceEvent, buggy_source: str) -> tuple[int | None, str]:
    line_no: int | None = None
    match = _LINE_NUMBER.search(event.content)
    if match is not None:
        parsed = int(match.group(1))
        if parsed > 0:
            line_no = parsed
    source = ""
    source_match = _SOURCE_FIELD.search(event.content)
    if source_match is not None:
        source = source_match.group(1).strip()
    if not source and line_no is not None and buggy_source:
        lines = buggy_source.rstrip("\n").splitlines()
        if 1 <= line_no <= len(lines):
            source = lines[line_no - 1].strip()
    return line_no, source


def line_text(events: tuple[EvidenceEvent, ...], buggy_source: str = "") -> str:
    """One-line locator: the last executed statement, not I/O and not a gold REPLACE."""
    line_no, source = failing_span(events, buggy_source)
    if not source:
        return "unavailable"
    clipped = _clip_gist_line(source, 180)
    if line_no is not None:
        return f"failing statement (line {line_no}): {clipped}"
    return f"failing statement: {clipped}"


def collated_trace_text(
    buggy_source: str,
    events: tuple[EvidenceEvent, ...],
    *,
    width: int = 120,
) -> str:
    """Haque collated analogue: last runtime note per source line as a comment.

    I/O events stay as a short header. Canonical source is never used.
    """
    notes: dict[int, str] = {}
    io_lines: list[str] = []
    for event in events:
        if event.role_id in IO_ROLES:
            io_lines.append(f"{ROLE_NAMES[event.role_id]}: {_clip_gist_line(event.content, width)}")
            continue
        match = _LINE_NUMBER.search(event.content)
        if match is None:
            continue
        notes[int(match.group(1))] = _clip_gist_line(event.content, width)
    source_lines = buggy_source.rstrip("\n").splitlines() or [""]
    collated = []
    for index, line in enumerate(source_lines, start=1):
        note = notes.get(index)
        collated.append(f"{line}  # TRACE: {note}" if note else line)
    body = "\n".join(collated)
    if io_lines:
        return "\n".join(io_lines) + "\n" + body
    return body or "unavailable"


def is_wrong_value_failure(example: RepairExample) -> bool:
    """Public failure is a wrong value, not an immediate exception."""
    fails = [event for event in example.events if event.role_id == FAIL]
    if not fails:
        return False
    blob = fails[-1].content.lower()
    if "error" in blob or "exception" in blob:
        return False
    return "fail" in blob


def shared_affix_edit_mask(buggy_ids: list[int], target_ids: list[int]) -> list[bool]:
    """True on target tokens outside the shared prefix/suffix with the buggy ids."""
    if not target_ids:
        return []
    prefix = 0
    limit = min(len(buggy_ids), len(target_ids))
    while prefix < limit and buggy_ids[prefix] == target_ids[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and buggy_ids[len(buggy_ids) - 1 - suffix] == target_ids[len(target_ids) - 1 - suffix]
    ):
        suffix += 1
    start, end = prefix, len(target_ids) - suffix
    if start >= end:
        return [True] * len(target_ids)
    return [start <= index < end for index in range(len(target_ids))]


def _source_lines_for_sketch(source: str) -> list[str]:
    try:
        return ast.unparse(ast.parse(source)).splitlines()
    except SyntaxError:
        return source.rstrip().splitlines()


def edit_sketch(buggy_source: str, target_source: str, *, max_ops: int = 8, width: int = 160) -> str:
    """Short oracle diagnosis: REPLACE/DELETE/INSERT lines, not the full canonical program."""
    buggy = _source_lines_for_sketch(buggy_source)
    target = _source_lines_for_sketch(target_source)
    if buggy == target:
        return "NO_EDIT"
    ops: list[str] = []
    matcher = difflib.SequenceMatcher(a=buggy, b=target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            ops.append(
                f"REPLACE `{_clip_gist_line(buggy[i1].strip(), width)}` "
                f"WITH `{_clip_gist_line(target[j1].strip(), width)}`"
            )
            continue
        if tag in {"replace", "delete"}:
            ops.extend(
                f"DELETE `{_clip_gist_line(line.strip(), width)}`" for line in buggy[i1:i2]
            )
        if tag in {"replace", "insert"}:
            ops.extend(
                f"INSERT `{_clip_gist_line(line.strip(), width)}`" for line in target[j1:j2]
            )
    if not ops:
        return "NO_EDIT"
    return "\n".join(ops[:max_ops])


def render_gist_user_prompt(example: RepairExample) -> str:
    return f"""The Python program below failed. Runtime evidence is spliced at the marker. Write only a short debugger note with the failing assertion, the observed outcome, and the last line or return. Do not repair the program.

BUGGY PROGRAM:
```python
{example.buggy_source.rstrip()}
```

RUNTIME EVIDENCE:
{MARKER}
"""


def render_gist_chat(tokenizer, example: RepairExample) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise Python debugger and program repair assistant.",
            },
            {"role": "user", "content": render_gist_user_prompt(example)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_sketch_user_prompt(example: RepairExample) -> str:
    return f"""The Python program below failed. Runtime evidence is spliced at the marker. Write only one REPLACE/DELETE/INSERT line naming the exact source change. Do not repair the program.

BUGGY PROGRAM:
```python
{example.buggy_source.rstrip()}
```

RUNTIME EVIDENCE:
{MARKER}
"""


def render_sketch_chat(tokenizer, example: RepairExample) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise Python debugger and program repair assistant.",
            },
            {"role": "user", "content": render_sketch_user_prompt(example)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_icae_encode_text(example: RepairExample, events: tuple[EvidenceEvent, ...]) -> str:
    """Encoder-only context: buggy program, public test, and a short runtime gist. No canonical patch."""
    return f"""Summarize this failing run into silent memory slots.

BUGGY PROGRAM:
```python
{example.buggy_source.rstrip()}
```

PUBLIC FAILING TEST:
`{example.failing_public_test}`

RUNTIME:
{gist_text(events)}
"""


def render_icae_encode_chat(tokenizer, example: RepairExample, events: tuple[EvidenceEvent, ...]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": render_icae_encode_text(example, events)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_user_prompt(
    example: RepairExample,
    evidence: str,
    *,
    include_specification: bool = False,
    include_public_test: bool = True,
    evidence_kind: str = "runtime",
) -> str:
    specification = (
        f"\nSpecification: {example.description}\n"
        if include_specification
        else ""
    )
    public_test = (
        f"\nPUBLIC FAILING TEST:\n{format_public_test(example.failing_public_test)}\n"
        if include_public_test
        else ""
    )
    if evidence_kind == "oracle":
        instruction = (
            "Repair the Python program below. An oracle edit sketch names the exact source "
            "change. Apply that change and return only the complete corrected program in one "
            "Python code block. Do not explain your reasoning."
        )
        evidence_heading = "ORACLE EDIT SKETCH"
    else:
        instruction = (
            "Repair the Python program below. Use the runtime evidence as debugger output.\n\n"
            "First identify the behavioral discrepancy from the ordered runtime events. Then return only\n"
            "the complete corrected program in one Python code block. Do not explain your reasoning."
        )
        evidence_heading = "RUNTIME EVIDENCE"
    return f"""{instruction}
{specification}
BUGGY PROGRAM:
```python
{example.buggy_source.rstrip()}
```
{public_test}
{evidence_heading}:
{evidence}
"""


def render_chat(
    tokenizer,
    example: RepairExample,
    evidence: str,
    *,
    include_specification: bool = False,
    include_public_test: bool = True,
    evidence_kind: str = "runtime",
) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise Python debugger and program repair assistant.",
            },
            {
                "role": "user",
                "content": render_user_prompt(
                    example,
                    evidence,
                    include_specification=include_specification,
                    include_public_test=include_public_test,
                    evidence_kind=evidence_kind,
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def evidence_for_condition(example: RepairExample, condition: str) -> str:
    if condition == "no_evidence":
        return "unavailable"
    if condition in {
        "true_latent",
        "shuffled_latent",
        "corrupted_latent",
        "discrepancy_embed",
        "shuffled_embed",
        "corrupted_embed",
    }:
        return MARKER
    if condition == "io_text":
        return compact_trace_text(io_only_events(example.events))
    if condition == "compact_text":
        return compact_trace_text(example.events)
    if condition == "gist_text":
        return gist_text(example.events)
    if condition == "discrepancy_text":
        return discrepancy_text(example.events)
    if condition == "line_text":
        return line_text(example.events, example.buggy_source)
    if condition == "collated_text":
        return collated_trace_text(example.buggy_source, example.events)
    if condition == "corrupted_text":
        return compact_trace_text(corrupt_runtime_events(example.events))
    if condition == "oracle_sketch":
        return edit_sketch(example.buggy_source, example.target_source)
    raise ValueError(f"unknown condition: {condition}")


def events_for_condition(
    example: RepairExample, condition: str, shuffled: RepairExample | None = None
) -> tuple[EvidenceEvent, ...]:
    if condition == "true_latent":
        return example.events
    if condition == "shuffled_latent":
        if shuffled is None:
            raise ValueError("shuffled_latent requires another example")
        return shuffled.events
    if condition == "corrupted_latent":
        return corrupt_runtime_events(example.events)
    raise ValueError(f"condition has no latent events: {condition}")


def allowed_calls_for_tests(tests: Iterable[str]) -> set[str]:
    """Contest stdio programs use input() and eval(input()); MBPP asserts do not."""
    if any(parse_stdio_test(test) is not None for test in tests):
        return {"input", "eval"}
    return set()


def extract_patch(response: str, *, allowed_calls: Iterable[str] = ()) -> str:
    allowed = set(allowed_calls)
    forbidden = FORBIDDEN_CALLS - allowed
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    for candidate in blocks or [response]:
        source = candidate.strip() + "\n"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or node.names[0].name).split(".")[0]]
            else:
                modules = []
            for module in modules:
                if module not in SAFE_IMPORTS:
                    raise ValueError(f"unsafe import: {module}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden
            ):
                raise ValueError(f"unsafe call: {node.func.id}")
        return source
    raise ValueError("no Python source found")


def evaluate_hidden(example: RepairExample, source: str, *, timeout_seconds: float = 3.0) -> dict:
    task = MBPPTask(
        task_id=example.task_id,
        description=example.description,
        canonical_source=source,
        tests=example.hidden_tests,
        setup_source=example.setup_source,
    )
    result = run_mbpp_tests(task, source, timeout_seconds=timeout_seconds)
    result["passed"] = result.get("status") == "all_pass"
    return result


def example_from_mutation(
    task: MBPPTask,
    mutation: MBPPMutation,
    *,
    plus_assertions: tuple[str, ...] = (),
    max_events_per_test: int = 24,
    timeout_seconds: float = 2.0,
) -> tuple[RepairExample | None, dict]:
    canonical = run_mbpp_tests(task, timeout_seconds=timeout_seconds)
    mutant = run_mbpp_tests(task, mutation.source, timeout_seconds=timeout_seconds)
    audit = {
        "task_id": task.task_id,
        "split": task.split,
        "mutation_kind": mutation.kind,
        "mutation_ordinal": mutation.ordinal,
        "source_hash": source_hash(mutation.source),
        "canonical_status": canonical.get("status"),
        "mutant_status": mutant.get("status"),
        "selected": False,
        "reason": "unclassified",
    }
    if canonical.get("status") != "all_pass":
        audit["reason"] = f"canonical_{canonical.get('status')}"
        return None, audit
    partition = partition_tests(
        task.tests, canonical.get("tests") or [], mutant.get("tests") or [], source=task.canonical_source
    )
    if partition is None:
        audit["reason"] = "no_disjoint_hidden_failure"
        return None, audit
    extras = extra_hidden_tests(task, partition, plus_assertions, timeout_seconds=timeout_seconds)
    hidden_tests = partition.hidden_tests + extras
    hidden_probe = replace(task, tests=hidden_tests)
    hidden_result = run_mbpp_tests(hidden_probe, mutation.source, timeout_seconds=timeout_seconds)
    if hidden_result.get("status") == "all_pass":
        audit["reason"] = "hidden_all_pass_after_plus"
        return None, audit
    events = build_events(
        task, mutation, partition.public_tests, max_events_per_test=max_events_per_test
    )
    if not events:
        audit["reason"] = "public_trace_failed"
        return None, audit
    public_result = run_mbpp_tests(
        replace(task, tests=partition.public_tests), mutation.source, timeout_seconds=timeout_seconds
    )
    example = RepairExample(
        example_id=f"{task.task_id}:{mutation.kind}:{mutation.ordinal}",
        task_id=task.task_id,
        split=task.split,
        mutation_kind=mutation.kind,
        mutation_ordinal=mutation.ordinal,
        description=task.description,
        buggy_source=mutation.source,
        target_source=task.canonical_source,
        setup_source=task.setup_source,
        failing_public_test=partition.public_tests[0],
        public_tests=partition.public_tests,
        hidden_tests=hidden_tests,
        public_hashes=partition.public_hashes,
        hidden_hashes=tuple(hash_key(key) for key in literal_calls(task.canonical_source, hidden_tests)),
        events=events,
        source_hash=source_hash(mutation.source),
        mutant_public_status=public_result.get("status", "unknown"),
        mutant_hidden_status=hidden_result.get("status", "unknown"),
    )
    audit.update(
        {
            "selected": True,
            "reason": "ok",
            "public_tests": list(example.public_tests),
            "hidden_tests": list(example.hidden_tests),
            "public_hashes": list(example.public_hashes),
            "hidden_hashes": list(example.hidden_hashes),
            "mutant_public_status": example.mutant_public_status,
            "mutant_hidden_status": example.mutant_hidden_status,
            "event_count": len(example.events),
        }
    )
    return example, audit


def _collect_one_task(
    task: MBPPTask,
    plus_assertions: tuple[str, ...],
    max_mutants: int,
    max_selected_per_task: int,
    max_events_per_test: int,
    timeout_seconds: float,
) -> tuple[list[RepairExample], list[dict]]:
    examples: list[RepairExample] = []
    audits: list[dict] = []
    try:
        mutations = generate_mutants(task, limit=max_mutants)
    except (SyntaxError, ValueError) as error:
        return [], [
            {
                "task_id": task.task_id,
                "split": task.split,
                "selected": False,
                "reason": f"mutation_error:{type(error).__name__}",
            }
        ]
    if not mutations:
        return [], [
            {
                "task_id": task.task_id,
                "split": task.split,
                "selected": False,
                "reason": "no_mutants",
            }
        ]
    selected = 0
    for mutation in mutations:
        example, audit = example_from_mutation(
            task,
            mutation,
            plus_assertions=plus_assertions,
            max_events_per_test=max_events_per_test,
            timeout_seconds=timeout_seconds,
        )
        audits.append(audit)
        if example is None or selected >= max_selected_per_task:
            continue
        examples.append(example)
        selected += 1
    return examples, audits


def collect_examples(
    tasks: tuple[MBPPTask, ...],
    *,
    plus_by_id: dict[int, tuple[str, ...]] | None = None,
    max_mutants: int = 12,
    max_selected_per_task: int = 1,
    max_events_per_test: int = 24,
    timeout_seconds: float = 2.0,
    workers: int = 24,
) -> tuple[list[RepairExample], list[dict]]:
    plus_by_id = plus_by_id or {}

    def process(task: MBPPTask) -> tuple[list[RepairExample], list[dict]]:
        return _collect_one_task(
            task,
            plus_by_id.get(task.task_id, ()),
            max_mutants,
            max_selected_per_task,
            max_events_per_test,
            timeout_seconds,
        )

    examples: list[RepairExample] = []
    audits: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_examples, task_audits in pool.map(process, tasks):
            examples.extend(task_examples)
            audits.extend(task_audits)
    return examples, audits


def example_to_json(example: RepairExample) -> dict:
    payload = asdict(example)
    payload["events"] = [asdict(event) for event in example.events]
    return payload


def example_from_json(row: dict) -> RepairExample:
    events = tuple(EvidenceEvent(**event) for event in row["events"])
    return RepairExample(
        example_id=row["example_id"],
        task_id=row["task_id"],
        split=row["split"],
        mutation_kind=row["mutation_kind"],
        mutation_ordinal=row["mutation_ordinal"],
        description=row["description"],
        buggy_source=row["buggy_source"],
        target_source=row["target_source"],
        setup_source=row["setup_source"],
        failing_public_test=row["failing_public_test"],
        public_tests=tuple(row["public_tests"]),
        hidden_tests=tuple(row["hidden_tests"]),
        public_hashes=tuple(row["public_hashes"]),
        hidden_hashes=tuple(row["hidden_hashes"]),
        events=events,
        source_hash=row["source_hash"],
        mutant_public_status=row["mutant_public_status"],
        mutant_hidden_status=row["mutant_hidden_status"],
    )


def read_examples(path: str | Path) -> list[RepairExample]:
    return [
        example_from_json(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def different_task_example(examples: list[RepairExample], index: int) -> RepairExample:
    task_id = examples[index].task_id
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.task_id != task_id:
            return candidate
    raise ValueError("shuffled control requires at least two task IDs")


def content_role_test(events: tuple[EvidenceEvent, ...]) -> tuple[list[str], list[int], list[int]]:
    return (
        [event.content for event in events],
        [event.role_id for event in events],
        [event.test_id for event in events],
    )
