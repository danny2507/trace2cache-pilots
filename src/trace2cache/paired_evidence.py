"""Matched A/B runtime evidence for the Trace2Cache ambiguity mechanism study.

The central invariant is deliberately stronger than ordinary supervised data: two views
share buggy code, inputs, runtime events, actual output and presentation order.  Only their
supplied expected values and the truthfully derived status may differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .ambiguous_repair import AmbiguousRepairCase, get_ambiguous_cases
from .structured_runtime import (
    StructuredTraceResult,
    canonical_value,
    execute_structured_trace,
    tagged_value,
    values_equal,
)


SCHEMA_VERSION = 2
TEST_START, INPUT, CALL, LINE, BRANCH, DEFINITION, RETURN, ACTUAL, EXPECTED, STATUS = range(10)
ROLE_NAMES = ("TEST_START", "INPUT", "CALL", "LINE", "BRANCH", "DEFINITION", "RETURN", "ACTUAL", "EXPECTED", "STATUS")


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: int
    test_id: int
    step: int
    role_id: int
    content: str
    source_line: int | None


@dataclass(frozen=True)
class EvidenceView:
    view_uid: str
    events: tuple[EvidenceEvent, ...]


@dataclass(frozen=True)
class PairedEvidenceRecord:
    schema_version: int
    pair_uid: str
    family_id: str
    split: str
    buggy_source: str
    function_name: str
    public_test: dict[str, Any]
    evidence_a: EvidenceView
    evidence_b: EvidenceView
    label_a: int
    label_b: int
    target_source_a: str
    target_source_b: str
    tests_metadata: tuple[dict[str, Any], ...]
    input_hash: str


@dataclass(frozen=True)
class InputDistribution:
    bundle_test_count: tuple[int, int]
    item_count: tuple[int, int]
    value_domain: str


PRIMARY_DISTRIBUTIONS: dict[str, InputDistribution] = {
    "train": InputDistribution((2, 4), (3, 5), "train"),
    "dev": InputDistribution((2, 4), (3, 5), "train"),
    "test_short": InputDistribution((2, 4), (3, 5), "train"),
    "test_bundle_long": InputDistribution((5, 8), (3, 5), "train"),
    "test_value_new": InputDistribution((2, 4), (3, 5), "new"),
    "test_bundle_long_value_new": InputDistribution((5, 8), (3, 5), "new"),
}


def expected_for(item: AmbiguousRepairCase, args: list[object]) -> object:
    """Independent simple reference semantics; checked against ``correct_source`` at build time."""
    values = args[0]
    case_id = item.case.case_id
    if case_id == "locate_first": return values.index(args[1])
    if case_id == "locate_last": return len(values) - 1 - list(reversed(values)).index(args[1])
    if case_id == "aggregate_sum": return sum(values)
    if case_id == "aggregate_product":
        result = 1
        for value in values: result *= value
        return result
    if case_id == "extreme_max": return max(values)
    if case_id == "extreme_min": return min(values)
    if case_id == "measure_even": return sum(value % 2 == 0 for value in values)
    if case_id == "measure_positive": return sum(value > 0 for value in values)
    if case_id == "arrange_ascending": return sorted(values)
    if case_id == "arrange_descending": return sorted(values, reverse=True)
    if case_id == "choose_shortest": return min(values, key=len)
    if case_id == "choose_longest": return max(values, key=len)
    if case_id == "truth_any_positive": return any(value > 0 for value in values)
    if case_id == "truth_all_positive": return all(value > 0 for value in values)
    if case_id == "unique_first": return list(dict.fromkeys(values))
    if case_id == "unique_last": return [value for index, value in enumerate(values) if value not in values[index + 1:]]
    if case_id == "median_lower": return sorted(values)[(len(values) - 1) // 2]
    if case_id == "median_upper": return sorted(values)[len(values) // 2]
    if case_id == "rotate_left": return values[1:] + values[:1]
    if case_id == "rotate_right": return values[-1:] + values[:-1]
    if case_id == "transform_absolute": return sum(abs(value) for value in values)
    if case_id == "transform_square": return sum(value * value for value in values)
    if case_id == "filter_even": return [value for value in values if value % 2 == 0]
    if case_id == "filter_positive": return [value for value in values if value > 0]
    raise ValueError(case_id)


def _numbers(rng: random.Random, count: int, domain: str) -> list[int]:
    if domain == "new":
        values = tuple(range(-20, -7)) + tuple(range(8, 21))
        return [rng.choice(values) for _ in range(count)]
    return [rng.randint(-4, 4) for _ in range(count)]


def sample_args(pair_id: str, rng: random.Random, distribution: InputDistribution) -> list[object]:
    """Draw one candidate input. Caller checks reference outputs before accepting it."""
    count = rng.randint(*distribution.item_count)
    domain = distribution.value_domain
    if pair_id == "locate":
        count = max(count, 3)
        target = rng.choice(tuple(range(-20, -7)) + tuple(range(8, 21))) if domain == "new" else rng.randint(-4, 4)
        values = _numbers(rng, count, domain)
        first, last = sorted(rng.sample(range(count), 2))
        values[first] = target; values[last] = target
        return [values, target]
    if pair_id == "choose":
        # Random contents eliminate the old 60-string-triple support bottleneck.
        alphabet = "abcdefghjkmnpqrstuvwxyz"
        lengths = rng.sample(range(1, 13 if domain == "new" else 7), min(3, count))
        return [["".join(rng.choice(alphabet) for _ in range(length)) for length in lengths]]
    if pair_id == "median":
        count = rng.choice((4, 6) if domain == "train" else (8, 10))
        pool = tuple(range(-20, -7)) + tuple(range(8, 21)) if domain == "new" else tuple(range(-8, 9))
        return [rng.sample(pool, count)]
    if pair_id == "truth":
        # Force mixed signs, which makes any-vs-all distinguish without relying on rejection.
        count = max(count, 2)
        values = _numbers(rng, count, domain)
        values[0] = abs(values[0]) or 1
        values[1] = -abs(values[1] or 1)
        return [values]
    if pair_id == "unique":
        count = max(count, 3)
        values = _numbers(rng, count, domain)
        values[0] = values[-1]
        if count > 2 and values[1] == values[0]: values[1] += 1
        return [values]
    if pair_id == "rotate":
        count = max(count, 3)
        values = _numbers(rng, count, domain)
        if len(set(values)) == 1: values[1] += 1
        return [values]
    return [_numbers(rng, count, domain)]


def _input_hash(source: str, bundle: list[list[object]]) -> str:
    normalized = {"source": source, "bundle": bundle}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _reference_output(item: AmbiguousRepairCase, args: list[object]) -> dict[str, Any]:
    result = execute_structured_trace(item.correct_source, item.case.function_name, args)
    if result.error is not None or result.output is None:
        raise AssertionError(f"reference execution failed for {item.case.case_id}: {result.error}")
    expected = tagged_value(expected_for(item, args))
    if not values_equal(result.output, expected):
        raise AssertionError(f"reference mismatch for {item.case.case_id}: {result.output} != {expected}")
    return expected


def _trace_role(event: Any, changed: bool) -> int:
    if event.event == "call": return CALL
    if event.event in {"return", "exception"}: return RETURN
    if event.source.startswith(("if ", "elif ", "for ", "while ")): return BRANCH
    return DEFINITION if changed else LINE


def _payload(value: Any) -> str:
    """Neutral deterministic payload; semantics enter through role metadata, not words."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _shared_test_events(args: list[object], trace: StructuredTraceResult, test_id: int) -> list[EvidenceEvent]:
    events = [EvidenceEvent(0, test_id, 0, TEST_START, _payload({"test": test_id}), None)]
    event_id = 1
    for arg_index, arg in enumerate(args):
        tagged = tagged_value(arg)
        events.append(EvidenceEvent(event_id, test_id, event_id, INPUT, _payload({"argument": arg_index, "value": tagged}), None)); event_id += 1
    previous: dict[str, Any] | None = None
    for trace_event in trace.events:
        changed = previous is None or trace_event.locals != previous
        payload = {"statement": trace_event.source, "state": trace_event.locals}
        if trace_event.value is not None: payload["value"] = trace_event.value
        events.append(EvidenceEvent(event_id, test_id, trace_event.step, _trace_role(trace_event, changed), _payload(payload), trace_event.line)); event_id += 1
        previous = trace_event.locals
    actual = trace.error if trace.error is not None else trace.output
    events.append(EvidenceEvent(event_id, test_id, event_id, ACTUAL, _payload(actual), None))
    return events


def _view_events(
    shared_by_test: list[list[EvidenceEvent]], expected: list[dict[str, Any]], statuses: list[str], order_expected_first: list[bool]
) -> tuple[EvidenceEvent, ...]:
    combined: list[EvidenceEvent] = []
    next_id = 0
    for test_id, shared in enumerate(shared_by_test):
        for event in shared:
            combined.append(EvidenceEvent(next_id, test_id, event.step, event.role_id, event.content, event.source_line)); next_id += 1
        actual_event = combined.pop()  # terminal shared ACTUAL is counterbalanced with expected.
        terminal = [actual_event, EvidenceEvent(next_id, test_id, actual_event.step, EXPECTED, _payload(expected[test_id]), None)]
        next_id += 1
        if order_expected_first[test_id]: terminal.reverse()
        for event in terminal:
            combined.append(EvidenceEvent(next_id, test_id, event.step, event.role_id, event.content, event.source_line)); next_id += 1
        combined.append(EvidenceEvent(next_id, test_id, actual_event.step, STATUS, _payload({"status": statuses[test_id]}), None)); next_id += 1
    return tuple(combined)


def _pair_cases() -> tuple[tuple[AmbiguousRepairCase, AmbiguousRepairCase], ...]:
    cases = get_ambiguous_cases()
    pairs = tuple((cases[index], cases[index + 1]) for index in range(0, len(cases), 2))
    for left, right in pairs:
        if (left.pair_id != right.pair_id or left.case.buggy_source != right.case.buggy_source or
                left.case.function_name != right.case.function_name or left.case.public_test != right.case.public_test):
            raise AssertionError(f"invalid ambiguous pair: {left.case.case_id}/{right.case.case_id}")
    return pairs


def generate_split(
    split: str,
    pairs_per_family: int,
    *,
    seed: int,
    distribution: InputDistribution | None = None,
    used_hashes: set[str] | None = None,
    max_attempts: int = 10_000,
) -> tuple[list[PairedEvidenceRecord], dict[str, int]]:
    """Create deterministic, disjoint pairs and count rejected candidates by family."""
    if pairs_per_family < 1: raise ValueError("pairs_per_family must be positive")
    distribution = distribution or PRIMARY_DISTRIBUTIONS[split]
    used_hashes = used_hashes if used_hashes is not None else set()
    rng = random.Random(seed)
    records: list[PairedEvidenceRecord] = []
    rejected: dict[str, int] = {}
    label_by_case = {item.case.case_id: index for index, item in enumerate(get_ambiguous_cases())}
    for family_index, (left, right) in enumerate(_pair_cases()):
        accepted = 0; rejected[ left.pair_id] = 0
        attempts = 0
        while accepted < pairs_per_family:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"exhausted {max_attempts} candidates for family {left.pair_id}")
            bundle = [sample_args(left.pair_id, rng, distribution) for _ in range(rng.randint(*distribution.bundle_test_count))]
            bundle_hash = _input_hash(left.case.buggy_source, bundle)
            if bundle_hash in used_hashes:
                rejected[left.pair_id] += 1; continue
            expected_a = [_reference_output(left, args) for args in bundle]
            expected_b = [_reference_output(right, args) for args in bundle]
            if not all(not values_equal(a, b) for a, b in zip(expected_a, expected_b)):
                rejected[left.pair_id] += 1; continue
            traces = [execute_structured_trace(left.case.buggy_source, left.case.function_name, args) for args in bundle]
            shared = [_shared_test_events(args, trace, test_id) for test_id, (args, trace) in enumerate(zip(bundle, traces))]
            actuals = [trace.error if trace.error is not None else trace.output for trace in traces]
            status_a = ["PASS" if values_equal(actual, expected) else "FAIL" for actual, expected in zip(actuals, expected_a)]
            status_b = ["PASS" if values_equal(actual, expected) else "FAIL" for actual, expected in zip(actuals, expected_b)]
            order = [bool(rng.getrandbits(1)) for _ in bundle]
            uid = f"{split}-{left.pair_id}-{accepted:04d}-{bundle_hash[:12]}"
            metadata = tuple({
                "args": args, "expected_a": expected_a[index], "expected_b": expected_b[index],
                "actual": actuals[index], "status_a": status_a[index], "status_b": status_b[index],
                "trace_truncated": traces[index].truncated, "terminal_expected_first": order[index],
            } for index, args in enumerate(bundle))
            records.append(PairedEvidenceRecord(
                SCHEMA_VERSION, uid, left.pair_id, split, left.case.buggy_source, left.case.function_name,
                {"args": left.case.public_test.args, "expected": tagged_value(left.case.public_test.expected)},
                EvidenceView(uid + "-a", _view_events(shared, expected_a, status_a, order)),
                EvidenceView(uid + "-b", _view_events(shared, expected_b, status_b, order)),
                label_by_case[left.case.case_id], label_by_case[right.case.case_id], left.correct_source, right.correct_source,
                metadata, bundle_hash,
            ))
            used_hashes.add(bundle_hash); accepted += 1
    return records, rejected


def serialize_record(record: PairedEvidenceRecord) -> dict[str, Any]:
    return asdict(record)


def _view_from_dict(raw: dict[str, Any]) -> EvidenceView:
    return EvidenceView(raw["view_uid"], tuple(EvidenceEvent(**event) for event in raw["events"]))


def deserialize_record(raw: dict[str, Any]) -> PairedEvidenceRecord:
    raw = dict(raw)
    raw["evidence_a"] = _view_from_dict(raw["evidence_a"]); raw["evidence_b"] = _view_from_dict(raw["evidence_b"])
    raw["tests_metadata"] = tuple(raw["tests_metadata"])
    return PairedEvidenceRecord(**raw)


def write_jsonl(records: Iterable[PairedEvidenceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(serialize_record(record), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[PairedEvidenceRecord]:
    with path.open(encoding="utf-8") as handle:
        return [deserialize_record(json.loads(line)) for line in handle if line.strip()]


def audit_records(records: Iterable[PairedEvidenceRecord]) -> dict[str, Any]:
    rows = list(records); errors: list[str] = []; hashes: set[str] = set(); counts: dict[str, int] = {}
    forbidden = ("diagnosis", "locate_first", "locate_last", "aggregate_sum", "aggregate_product")
    for record in rows:
        counts[record.family_id] = counts.get(record.family_id, 0) + 1
        if record.schema_version != SCHEMA_VERSION: errors.append(f"{record.pair_uid}: schema")
        if record.input_hash in hashes: errors.append(f"{record.pair_uid}: duplicate input hash")
        hashes.add(record.input_hash)
        if record.label_a == record.label_b or record.target_source_a == record.target_source_b: errors.append(f"{record.pair_uid}: non-opposite labels")
        if len(record.evidence_a.events) != len(record.evidence_b.events): errors.append(f"{record.pair_uid}: event length")
        for event_a, event_b in zip(record.evidence_a.events, record.evidence_b.events):
            if event_a.role_id not in (EXPECTED, STATUS) and event_a != event_b: errors.append(f"{record.pair_uid}: shared event changed"); break
            if any(word in event_a.content.lower() for word in forbidden): errors.append(f"{record.pair_uid}: forbidden lexical leakage"); break
        for meta in record.tests_metadata:
            if values_equal(meta["expected_a"], meta["expected_b"]): errors.append(f"{record.pair_uid}: indistinguishable test")
            true_a = "PASS" if values_equal(meta["actual"], meta["expected_a"]) else "FAIL"
            true_b = "PASS" if values_equal(meta["actual"], meta["expected_b"]) else "FAIL"
            if meta["status_a"] != true_a or meta["status_b"] != true_b: errors.append(f"{record.pair_uid}: false status")
    return {"schema_version": SCHEMA_VERSION, "records": len(rows), "families": counts, "input_hashes": len(hashes), "errors": errors, "passed": not errors}
