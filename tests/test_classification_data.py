"""Classification labels and foundation-model inputs must share a causal contract."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from build_precipitation_dataset import build_dataset
from classification_data import (
    CALENDAR_COLUMNS, build_classification_dataset, load_classification_bundle,
)
from precipitation_data import PrecipitationConfig, load_precipitation_bundle


class ClassificationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "hourly.csv"
        self.parent = self.root / "parent"
        self.output = self.root / "classification"
        times = pd.date_range("2019-12-28", "2020-01-06 23:00", freq="h", tz="UTC", name="datetime")
        self.raw = pd.DataFrame({
            "TT_TU": np.arange(len(times)) * .01, "RF_TU": 70.,
            "R1": .1, "SD_SO": 20., "F": 2., "P0": 980.,
        }, index=times)
        self.cfg = PrecipitationConfig(
            lookback_hours=6, train_end_exclusive="2020-01-01",
            validation_end_exclusive="2020-01-03", source_start=None,
            accepted_target_quality_levels=None,
        )

    def build(self, cfg=None, output=None):
        self.raw.to_csv(self.source)
        build_dataset(self.source, self.parent, cfg or self.cfg)
        build_classification_dataset(self.parent, output or self.output)
        return load_classification_bundle(output or self.output)

    def test_class_boundaries_and_shared_parent_cases_without_resampling(self):
        reference = pd.Timestamp("2020-01-04 12:00", tz="UTC")
        self.raw.loc[reference + pd.Timedelta(hours=1):reference + pd.Timedelta(hours=3), "R1"] = [0., 0., 0.]
        self.raw.loc[reference + pd.Timedelta(hours=4):reference + pd.Timedelta(hours=6), "R1"] = [2., .8, .1]
        self.raw.loc[reference + pd.Timedelta(hours=10):reference + pd.Timedelta(hours=12), "R1"] = [2.3, .4, .3]
        bundle = self.build()
        lookup = bundle.cases.merge(bundle.labels, on="case_id").set_index(["issue_reference_time", "horizon_h"])
        self.assertEqual(lookup.loc[(reference, 3), "target_class"], 0)
        self.assertEqual(lookup.loc[(reference, 6), "target_class"], 1)
        self.assertEqual(lookup.loc[(reference, 12), "target_class"], 2)
        parent = load_precipitation_bundle(self.parent)
        for horizon in self.cfg.horizons_hours:
            for split in ("train", "validation", "evaluation"):
                old_x, old_y, old_cases = parent.arrays(horizon, split)
                x, y, cases = bundle.arrays(horizon, split)
                flat, flat_y, flat_cases = bundle.arrays(horizon, split, flatten=True)
                np.testing.assert_array_equal(old_x, x)
                np.testing.assert_array_equal(old_y, y)
                np.testing.assert_array_equal(x.reshape(len(x), -1), flat)
                np.testing.assert_array_equal(y, flat_y)
                pd.testing.assert_frame_equal(cases, flat_cases)
                pd.testing.assert_series_equal(cases.issue_time, old_cases.issue_time, check_names=False)
                self.assertNotIn("target_class", cases.columns)
                self.assertNotIn("observed_target_mm", cases.columns)
        balance = json.loads((self.output / "class_balance.json").read_text())
        self.assertEqual(sum(x["n"] for x in balance), len(bundle.cases))

    def test_foundation_inputs_preserve_missing_rain_and_never_expose_labels(self):
        reference = pd.Timestamp("2020-01-04 12:00", tz="UTC")
        self.raw.loc[reference - pd.Timedelta(hours=2), "R1"] = np.nan
        self.raw.loc[reference - pd.Timedelta(hours=3):reference - pd.Timedelta(hours=1), "TT_TU"] = np.nan
        bundle = self.build()
        case = bundle.cases.loc[bundle.cases.issue_reference_time.eq(reference) & bundle.cases.horizon_h.eq(3)].iloc[0]
        record = next(r for r in bundle.foundation_records(3, "evaluation") if r["case_id"] == case.case_id)
        self.assertTrue(np.isnan(record["target"][-2]))
        self.assertEqual(record["past_covariates"]["R1_missing"][-2], 1.)
        self.assertEqual(record["past_covariates"]["TT_TU_carried"][-1], 1.)
        self.assertEqual(set(record["future_covariates"]), set(CALENDAR_COLUMNS))
        self.assertNotIn("R1", record["past_covariates"])
        self.assertNotIn("target_class", record)
        self.assertNotIn("observed_target_mm", record)
        self.assertEqual(record["context_timestamps"][-1], case.input_end)
        self.assertLess(record["context_timestamps"][-1], case.issue_time)
        observed = self.raw.loc[case.input_start:case.input_end, "R1"].to_numpy()
        np.testing.assert_array_equal(record["target"], observed)

    def test_forecast_gap_and_last_three_hours_for_all_horizons(self):
        # Including delay=0 makes the first returned reference timestamp later
        # than the last available reference, even though issuance is at :50.
        for delay in (0, 1, 3):
            bundle = self.build(replace(self.cfg, observation_delay_hours=delay))
            for horizon in self.cfg.horizons_hours:
                record = next(bundle.foundation_records(horizon, "evaluation"))
                case = bundle.select(horizon, "evaluation").iloc[0]
                self.assertEqual(record["prediction_length"], horizon + delay)
                target_times = record["future_timestamps"][record["class_target_hour_indices"]]
                expected = pd.date_range(case.target_reference_end - pd.Timedelta(hours=2), periods=3, freq="h")
                pd.testing.assert_index_equal(target_times, expected)
                self.assertEqual(target_times[-1] + pd.Timedelta(minutes=self.cfg.reference_offset_minutes), case.target_end_inclusive)
                self.assertEqual(record["future_timestamps"][0], case.input_end + pd.Timedelta(hours=1))

    def test_future_weather_changes_labels_but_not_prediction_inputs(self):
        first = self.build()
        record = next(first.foundation_records(3, "evaluation"))
        case = first.select(3, "evaluation").iloc[0]
        x, _, _ = first.arrays(3, "evaluation")
        # Change all physical weather values after this case's input cut-off.
        self.raw.loc[self.raw.index > case.input_end, "R1"] = 2.
        self.raw.loc[self.raw.index > case.input_end, "TT_TU"] = 40.
        second = self.build(output=self.root / "changed")
        changed = next(second.foundation_records(3, "evaluation"))
        changed_x, _, _ = second.arrays(3, "evaluation")
        np.testing.assert_array_equal(x[0], changed_x[0])
        np.testing.assert_array_equal(record["target"], changed["target"])
        for key in ("past_covariates", "future_covariates"):
            for col in record[key]:
                np.testing.assert_array_equal(record[key][col], changed[key][col])
        old_y = first.labels.set_index("case_id").loc[case.case_id, "target_class"]
        new_y = second.labels.set_index("case_id").loc[case.case_id, "target_class"]
        self.assertNotEqual(old_y, new_y)

    def test_missing_targets_stay_excluded_and_parent_is_unchanged(self):
        self.raw.iloc[-10, self.raw.columns.get_loc("R1")] = np.nan
        self.raw.to_csv(self.source)
        build_dataset(self.source, self.parent, self.cfg)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.parent.iterdir()}
        build_classification_dataset(self.parent, self.output)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.parent.iterdir()}
        self.assertEqual(before, after)
        parent = load_precipitation_bundle(self.parent)
        bundle = load_classification_bundle(self.output)
        self.assertEqual(len(bundle.cases), parent.samples.eligible.sum())
        self.assertFalse(bundle.labels.target_class.isna().any())
        self.assertTrue((~parent.samples.target_complete).any())
        with self.assertRaisesRegex(ValueError, "separate"):
            build_classification_dataset(self.parent, self.parent)

    def test_schema_tracks_custom_class_threshold_and_rejects_tampering(self):
        self.build(replace(self.cfg, upper_class_threshold_mm=.3))
        schema = json.loads((self.output / "schema.json").read_text())
        self.assertEqual(schema["classes"][1]["exclusive_max_mm"], .3)
        self.assertEqual(schema["classes"][2]["inclusive_min_mm"], .3)
        with (self.output / "schema.json").open("a") as f:
            f.write(" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_classification_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
