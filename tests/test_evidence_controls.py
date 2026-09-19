from __future__ import annotations

import json

from trace2cache.evidence_controls import compact_behavioral_text, corrupt_runtime_keep_io, permute_runtime_temporal_bindings, io_only, permute_runtime_test_bindings, structured_text
from trace2cache.paired_evidence import ACTUAL, BRANCH, EXPECTED, INPUT, STATUS, TEST_START, EvidenceEvent, EvidenceView


def test_stage_c_controls_preserve_io_and_only_corrupt_runtime() -> None:
    view = EvidenceView("view", (
        EvidenceEvent(0, 0, 0, TEST_START, "{}", None),
        EvidenceEvent(1, 0, 1, INPUT, '{"value":1}', None),
        EvidenceEvent(2, 0, 2, BRANCH, '{"statement":"if x"}', 7),
        EvidenceEvent(3, 0, 3, ACTUAL, '{"type":"int","value":0}', None),
        EvidenceEvent(4, 0, 4, EXPECTED, '{"type":"int","value":1}', None),
        EvidenceEvent(5, 0, 5, STATUS, '{"status":"fail"}', None),
    ))
    compact = io_only(view)
    assert [event.event_id for event in compact.events] == [0, 1, 3, 4, 5]
    corrupt = corrupt_runtime_keep_io(view)
    assert corrupt.events[1] == view.events[1]
    assert corrupt.events[3:] == view.events[3:]
    assert json.loads(corrupt.events[2].content) == {"runtime": "CORRUPTED"}
    assert corrupt.events[2].source_line is None
    assert '"role":"EXPECTED"' in structured_text(view)
    compact_text = compact_behavioral_text(view)
    assert "TEST 0" in compact_text
    assert "BRANCH step=2 line=7" in compact_text
    assert "EXPECTED step=4" in compact_text


def test_binding_permutation_preserves_io_rows_and_moves_same_role_runtime_packets() -> None:
    view = EvidenceView("two_tests", (
        EvidenceEvent(0, 0, 0, TEST_START, '{"test":0}', None),
        EvidenceEvent(1, 0, 1, INPUT, '{"value":1}', None),
        EvidenceEvent(2, 0, 2, BRANCH, '{"statement":"if x == 1"}', 7),
        EvidenceEvent(3, 0, 3, ACTUAL, '{"value":0}', None),
        EvidenceEvent(4, 0, 4, EXPECTED, '{"value":1}', None),
        EvidenceEvent(5, 0, 5, STATUS, '{"status":"fail"}', None),
        EvidenceEvent(6, 1, 0, TEST_START, '{"test":1}', None),
        EvidenceEvent(7, 1, 1, INPUT, '{"value":2}', None),
        EvidenceEvent(8, 1, 2, BRANCH, '{"statement":"if x == 2"}', 9),
        EvidenceEvent(9, 1, 3, ACTUAL, '{"value":0}', None),
        EvidenceEvent(10, 1, 4, EXPECTED, '{"value":2}', None),
        EvidenceEvent(11, 1, 5, STATUS, '{"status":"fail"}', None),
    ))
    permuted = permute_runtime_test_bindings(view)
    assert len(permuted.events) == len(view.events)
    for original, changed in zip(view.events, permuted.events):
        assert (original.event_id, original.test_id, original.step, original.role_id) == (
            changed.event_id, changed.test_id, changed.step, changed.role_id
        )
        if original.role_id in (TEST_START, INPUT, ACTUAL, EXPECTED, STATUS):
            assert original == changed
    assert permuted.events[2].content == view.events[8].content
    assert permuted.events[2].source_line == view.events[8].source_line
    assert permuted.events[8].content == view.events[2].content


def test_temporal_permutation_preserves_each_test_runtime_packet_multiset() -> None:
    view = EvidenceView("one_test", (
        EvidenceEvent(0, 0, 0, TEST_START, '{"test":0}', None),
        EvidenceEvent(1, 0, 1, INPUT, '{"value":1}', None),
        EvidenceEvent(2, 0, 2, BRANCH, '{"statement":"if x"}', 7),
        EvidenceEvent(3, 0, 3, BRANCH, '{"statement":"return x"}', 8),
        EvidenceEvent(4, 0, 4, ACTUAL, '{"value":0}', None),
        EvidenceEvent(5, 0, 5, EXPECTED, '{"value":1}', None),
        EvidenceEvent(6, 0, 6, STATUS, '{"status":"fail"}', None),
    ))
    permuted = permute_runtime_temporal_bindings(view)
    original_packets = sorted((event.content, event.source_line) for event in view.events if event.role_id not in (TEST_START, INPUT, ACTUAL, EXPECTED, STATUS))
    changed_packets = sorted((event.content, event.source_line) for event in permuted.events if event.role_id not in (TEST_START, INPUT, ACTUAL, EXPECTED, STATUS))
    assert changed_packets == original_packets
    assert permuted.events[2].content == view.events[3].content
    for original, changed in zip(view.events, permuted.events):
        assert (original.event_id, original.test_id, original.step, original.role_id) == (changed.event_id, changed.test_id, changed.step, changed.role_id)
        if original.role_id in (TEST_START, INPUT, ACTUAL, EXPECTED, STATUS): assert original == changed
