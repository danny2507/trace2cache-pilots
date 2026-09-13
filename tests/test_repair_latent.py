import unittest

from trace2cache.repair_latent import changed_locals, select_temporal_events, summarize_events
from trace2cache.runtime import execute_with_trace


class RepairLatentTests(unittest.TestCase):
    def test_changed_locals_includes_deletion(self):
        self.assertEqual(
            changed_locals({"x": "1", "old": "2"}, {"x": "3"}),
            {"old": ("2", None), "x": ("1", "3")},
        )

    def test_temporal_selection_keeps_endpoints(self):
        summaries = [str(index) for index in range(20)]
        selected = select_temporal_events(summaries, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], "0")
        self.assertEqual(selected[-1], "19")

    def test_trace_summary_contains_state_and_return(self):
        trace = execute_with_trace("def f(x):\n    y = x + 1\n    return y\n", "f", [2])
        summaries = summarize_events(trace.events)
        self.assertTrue(any("y: None -> 3" in summary for summary in summaries))
        self.assertTrue(any("outcome=3" in summary for summary in summaries))


if __name__ == "__main__":
    unittest.main()
