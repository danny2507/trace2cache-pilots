from __future__ import annotations

import random
from types import SimpleNamespace
import unittest

from trace2cache.training_sampling import sample_paired_family


class TrainingSamplingTest(unittest.TestCase):
    def test_objective_forks_can_replay_the_same_pair_sequence(self):
        records = [SimpleNamespace(family_id=family, pair_uid=f"{family}-{index}")
                   for family in ("a", "b", "c") for index in range(4)]
        left, right = random.Random(418), random.Random(418)
        for _ in range(10):
            first = sample_paired_family(records, left, 8)
            second = sample_paired_family(records, right, 8)
            self.assertEqual([row.pair_uid for row in first], [row.pair_uid for row in second])
            self.assertEqual(len({row.family_id for row in first}), 1)

    def test_rejects_empty_or_zero_pair_sampling(self):
        with self.assertRaises(ValueError):
            sample_paired_family([], random.Random(1), 1)
        with self.assertRaises(ValueError):
            sample_paired_family([SimpleNamespace(family_id="a")], random.Random(1), 0)
