from __future__ import annotations

import json

from trace2cache.evidence_controls import compact_behavioral_text, corrupt_runtime_keep_io, io_only, structured_text
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
