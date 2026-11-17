import unittest

import numpy as np
import pandas as pd

from evaluate_precipitation_baselines import (
    fit_count_baselines, latest_observed_class, predict_count_baselines, probability_metrics,
)


class PrecipitationBaselineTests(unittest.TestCase):
    def test_missing_history_has_unknown_state_and_persistence_prior(self):
        times = pd.date_range("2020-01-01", periods=6, freq="h")
        observations = pd.DataFrame({"R1": [0., np.nan, 0., 0., 0., 0.]}, index=times)
        cases = pd.DataFrame({"input_end": [times[2], times[5]],
                              "target_end_inclusive": [times[2], times[5]]})
        states = latest_observed_class(observations, cases, 3.)
        np.testing.assert_array_equal(states, [3, 0])
        fitted = {"prior": np.array([.7, .2, .1]), "transitions": np.tile([.7, .2, .1], (4, 1)),
                  "seasonal": np.tile([.7, .2, .1], (12, 24, 1))}
        p = predict_count_baselines(fitted, cases, states)["last_observed_class"]
        np.testing.assert_array_equal(p[0], fitted["prior"])
        np.testing.assert_array_equal(p[1], [1., 0., 0.])

    def test_fit_needs_only_training_and_empty_cells_use_prior(self):
        training = pd.DataFrame({"target_class": [0, 1, 2],
                                 "target_end_inclusive": pd.date_range("2020-01-01", periods=3, freq="3h")})
        fitted = fit_count_baselines(training, np.array([0, 0, 1]))
        np.testing.assert_array_equal(fitted["seasonal"][11, 23], fitted["prior"])
        np.testing.assert_array_equal(fitted["transitions"][3], fitted["prior"])
        self.assertTrue(np.allclose(fitted["transitions"].sum(axis=1), 1))

    def test_scores_distinguish_accuracy_from_probability_quality(self):
        y = np.array([0, 0, 1, 2])
        dry = np.tile([1., 0., 0.], (4, 1))
        calibrated_frequency = np.tile([.5, .25, .25], (4, 1))
        a = probability_metrics(y, dry)
        b = probability_metrics(y, calibrated_frequency)
        self.assertEqual(a["accuracy"], b["accuracy"])
        self.assertLess(b["brier_multiclass"], a["brier_multiclass"])
        self.assertEqual(a["class2_recall"], 0)
        self.assertAlmostEqual(a["brier_multiclass"], 2 * (1 - a["accuracy"]))
        with self.assertRaises(ValueError):
            probability_metrics(y, dry * 2)


if __name__ == "__main__":
    unittest.main()
