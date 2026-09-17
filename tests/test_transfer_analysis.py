import unittest
from types import SimpleNamespace

from trace2cache.transfer_analysis import analyze_interventions, audit_input_overlap, fully_novel_pair_uids, failure_categories


class TransferAnalysisTest(unittest.TestCase):
    def rows(self):
        return [{"family": family, "pair_uid": family + "-0", "side": side, "condition": condition,
                 "intended": {"passed": condition == "true_latent"},
                 "opposite": {"passed": condition == "paired_swap"}}
                for family in ("f1", "f2") for side in ("a", "b")
                for condition in ("true_latent", "paired_swap")]

    def test_perfect_paired_control_and_determinism(self):
        result = analyze_interventions(self.rows(), resamples=20)
        self.assertEqual(result, analyze_interventions(self.rows(), resamples=20))
        self.assertEqual(result["metrics"]["true_minus_swap"], 1.0)
        self.assertEqual(result["cluster_bootstrap_95pct"]["true_pair_success"], [1.0, 1.0])
        self.assertTrue(result["all_numerical_gates_pass"])
        self.assertEqual(result["program_clusters"], 2)

    def test_incomplete_pair_and_duplicate_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_interventions(self.rows()[:-1])
        with self.assertRaises(ValueError):
            analyze_interventions(self.rows() + self.rows()[:1])

    def test_correct_code_identity_does_not_count_as_repair(self):
        rows = self.rows()
        for row in rows:
            row["nearest_label"] = row["label"] = 0
        rows[0]["intended"]["passed"] = False
        metrics = analyze_interventions(rows, resamples=20)["metrics"]
        self.assertEqual(metrics["true_nearest_code_accuracy"], 1.0)
        self.assertEqual(metrics["true_repair"], 0.75)
        self.assertEqual(metrics["correct_nearest_but_failed_repair"], 0.25)

    def test_sandbox_rejection_is_separate_from_failed_execution(self):
        rows = [{"intended": {"passed": False, "error": "ValueError: call is not allowlisted: set"}},
                {"intended": {"passed": False, "tests": [{"passed": False}]}},
                {"intended": {"passed": False, "error": "ValueError: no parseable function named f"}}]
        self.assertEqual(failure_categories(rows), {"sandbox_policy_rejected": 1, "executed_test_failure": 1, "parse_rejected": 1})

    def test_exact_bundle_disjointness_does_not_imply_individual_input_disjointness(self):
        def record(uid, values):
            return SimpleNamespace(pair_uid=uid, input_hash=uid, buggy_source="program",
                                   tests_metadata=[{"args": [value]} for value in values])
        train = [record("train", [1, 2])]
        dev = [record("dev1", [2, 3]), record("dev2", [4, 5])]
        audit = audit_input_overlap(train, dev)
        self.assertEqual(audit["exact_bundle_hash_overlap"], 0)
        self.assertEqual(audit["individual_test_overlap"], 1)
        self.assertEqual(audit["development_pairs_with_no_individual_test_overlap"], 1)
        self.assertEqual(fully_novel_pair_uids(train, dev), {"dev2"})
