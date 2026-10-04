"""Check confidence interval calculations and missing-seed handling."""
import math
import unittest
from plot_ci import seed_interval, collect, ROOT


class IntervalTests(unittest.TestCase):
    def test_known_three_seed_interval(self):
        result = seed_interval([1, 2, 3])
        half_width = 4.302652729911275 / math.sqrt(3)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["degrees_of_freedom"], 2)
        self.assertAlmostEqual(result["lower"], 2 - half_width, places=9)
        self.assertAlmostEqual(result["upper"], 2 + half_width, places=9)

    def test_missing_seed_does_not_create_two_seed_ci(self):
        result = seed_interval([.5, None, 0])
        self.assertEqual(result["available_seeds"], 2)
        for key in ("mean", "lower", "upper", "standard_error", "degrees_of_freedom"):
            self.assertIsNone(result[key])

    def test_zero_variance(self):
        result = seed_interval([0, 0, 0])
        self.assertEqual((result["mean"], result["lower"], result["upper"]), (0, 0, 0))

    def test_intervals_are_not_clipped(self):
        self.assertGreater(seed_interval([0, 50, 100])["upper"], 100)
        self.assertLess(seed_interval([0, 50, 100])["lower"], 0)

    def test_nonfinite_or_wrong_sample_size_rejected(self):
        for values in ([1, 2], [1, 2, 3, 4], [1, float("nan"), 2], [1, float("inf"), 2]):
            with self.assertRaises(ValueError):
                seed_interval(values)

    def test_real_forecasts_and_pairing(self):
        report, inputs = collect(ROOT)
        self.assertEqual(report["verified_forecast_draws"], 720)
        self.assertEqual(len(inputs), 25)
        item = report["models"]["qwen25_7b"]["chatml"]
        self.assertEqual(item["paired_valid"], [5, 7, 4])
        result = item["deadlines"]["2035"]["paired_change_ci"]
        self.assertAlmostEqual(result["mean"], (-20 - 12 - 5) / 3)
        self.assertLess(result["lower"], result["mean"])
        self.assertGreater(result["upper"], 0)
        for year in report["models"]["qwen3_8b"]["chatml"]["deadlines"].values():
            self.assertIsNone(year["paired_change_ci"]["mean"])
            self.assertIsNone(year["trained_probability_ci"]["mean"])


if __name__ == "__main__":
    unittest.main()
