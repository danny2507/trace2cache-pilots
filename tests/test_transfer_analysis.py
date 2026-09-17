import unittest

from trace2cache.transfer_analysis import analyze_interventions


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
