from __future__ import annotations

from pathlib import Path
import unittest

from trace2cache.refactory import load_refactory_q1


class RefactoryTests(unittest.TestCase):
    def test_loads_real_failing_submission(self):
        root = Path("third_party/refactory")
        if not root.exists():
            self.skipTest("optional Refactory checkout is unavailable")
        case = load_refactory_q1(root, limit=1)[0]
        self.assertEqual(case.case_id, "refactory_q1_wrong_1_001")
        self.assertEqual(case.function_name, "search")
        self.assertEqual(len(case.hidden_tests), 10)


if __name__ == "__main__":
    unittest.main()
