from __future__ import annotations

import unittest

import torch

from trace2cache.discrepancy_residual import DiscrepancyResidual
from trace2cache.mbpp_generalization import (
    FAIL,
    TEST,
    EvidenceEvent,
    discrepancy_fields,
    discrepancy_text,
)


class DiscrepancyResidualTests(unittest.TestCase):
    def test_different_io_moves_slots(self):
        torch.manual_seed(0)
        encoder = DiscrepancyResidual(8, num_slots=4, mlp_width=16)
        expected = torch.randn(1, 8)
        got = torch.randn(1, 8)
        other = torch.randn(1, 8)
        true_slots = encoder.compose(encoder.delta_vector(expected, got))
        shuffled = encoder.compose(encoder.delta_vector(other, got))
        self.assertEqual(tuple(true_slots.shape), (1, 4, 8))
        self.assertFalse(torch.allclose(true_slots, shuffled, atol=1e-5))

    def test_offsets_start_at_zero_so_slots_match_within_example(self):
        encoder = DiscrepancyResidual(8, num_slots=4, mlp_width=16)
        self.assertTrue(torch.equal(encoder.slot_offsets, torch.zeros_like(encoder.slot_offsets)))
        expected = torch.randn(1, 8)
        got = torch.randn(1, 8)
        slots = encoder.compose(encoder.delta_vector(expected, got))
        self.assertTrue(torch.allclose(slots[0, 0], slots[0, 3]))

    def test_compose_is_unit_rms_before_scale(self):
        encoder = DiscrepancyResidual(8, num_slots=4, mlp_width=16)
        with torch.no_grad():
            encoder.scale.fill_(1.0)
        expected = torch.randn(2, 8)
        got = torch.randn(2, 8)
        slots = encoder.compose(encoder.delta_vector(expected, got))
        rms = slots.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_probe_loss_is_finite(self):
        encoder = DiscrepancyResidual(8, num_slots=4, mlp_width=16)
        expected = torch.randn(1, 8)
        got = torch.randn(1, 8)
        slots = encoder.compose(encoder.delta_vector(expected, got))
        loss = encoder.probe_loss(slots, expected, got)
        self.assertTrue(torch.isfinite(loss))

    def test_discrepancy_fields_are_swappable(self):
        events = (
            EvidenceEvent(0, TEST, "test stdin='1' expected='4'"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 4; got 6"),
        )
        expected, got = discrepancy_fields(events)
        self.assertEqual(expected.strip(), "4")
        self.assertEqual(got.strip(), "6")

    def test_discrepancy_fields_use_error_when_got_is_missing(self):
        events = (
            EvidenceEvent(0, TEST, "test stdin='122' expected='41578000\\n'"),
            EvidenceEvent(0, FAIL, "test outcome error:RuntimeError: expected 41578000; error RuntimeError"),
        )
        expected, got = discrepancy_fields(events)
        self.assertEqual(expected.strip(), "41578000")
        self.assertEqual(got.strip(), "RuntimeError")

    def test_discrepancy_text_swaps_expected_and_got(self):
        events = (
            EvidenceEvent(0, TEST, "test stdin='1' expected='4'"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 4; got 6"),
        )
        self.assertEqual(discrepancy_text(events), "DISCREPANCY: expected 4 got 6")
        self.assertEqual(discrepancy_text(events, swap=True), "DISCREPANCY: expected 6 got 4")


if __name__ == "__main__":
    unittest.main()
