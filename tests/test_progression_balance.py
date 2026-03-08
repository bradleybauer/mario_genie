import unittest

from mario_world_model.coverage import compute_progression_balance


class ProgressionBalanceTests(unittest.TestCase):
    def test_initial_support_seeds_bin_zero_for_all_levels(self) -> None:
        report = compute_progression_balance(
            progression_cov={},
            reachable_bins=set(),
            all_levels=[(1, 1), (1, 2)],
            bin_size=100,
        )

        self.assertEqual(report.total_frames, 0)
        self.assertEqual(report.num_bins, 2)
        self.assertEqual(set(report.weights.keys()), {(1, 1, 0), (1, 2, 0)})
        self.assertAlmostEqual(report.weights[(1, 1, 0)], 0.5)
        self.assertAlmostEqual(report.weights[(1, 2, 0)], 0.5)

    def test_unsupported_observed_bins_do_not_receive_weight(self) -> None:
        report = compute_progression_balance(
            progression_cov={
                (1, 1, 0): 100,
                (1, 1, 1): 20,
                (1, 1, 2): 5,
            },
            reachable_bins={(1, 1, 1)},
            all_levels=[(1, 1)],
            bin_size=100,
        )

        self.assertEqual(set(report.weights.keys()), {(1, 1, 0), (1, 1, 1)})
        self.assertNotIn((1, 1, 2), report.weights)
        self.assertGreater(report.weights[(1, 1, 0)], 0.0)
        self.assertGreater(report.weights[(1, 1, 1)], 0.0)

    def test_saturated_start_bins_keep_strong_exploration_weight(self) -> None:
        report = compute_progression_balance(
            progression_cov={
                (1, 1, 0): 1000,
                (1, 1, 1): 0,
            },
            reachable_bins={(1, 1, 1)},
            all_levels=[(1, 1)],
            bin_size=100,
        )

        self.assertLess(report.weights[(1, 1, 0)], 0.01)
        self.assertGreater(report.weights[(1, 1, 1)], 0.99)

    def test_unseen_stage_start_gets_gateway_bonus(self) -> None:
        report = compute_progression_balance(
            progression_cov={
                (8, 4, 20): 0,
                (8, 4, 21): 0,
                (8, 4, 22): 0,
                (8, 4, 39): 0,
            },
            reachable_bins={(8, 4, 20), (8, 4, 21), (8, 4, 22), (8, 4, 39)},
            all_levels=[(6, 4), (8, 1), (8, 4)],
            bin_size=100,
        )

        self.assertGreater(report.weights[(6, 4, 0)], report.weights[(8, 4, 20)])
        self.assertGreater(report.weights[(8, 1, 0)], report.weights[(8, 4, 20)])


if __name__ == "__main__":
    unittest.main()