"""Scientific data-contract checks: targets, causal inputs and temporal splits."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_precipitation_dataset import build_dataset
from precipitation_data import (
    WEATHER_COLUMNS, PrecipitationBundle, PrecipitationConfig, build_samples,
    load_precipitation_bundle, prepare_features, read_observations,
)


class PrecipitationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "hourly.csv"
        index = pd.date_range("2019-12-28", "2020-01-06 23:00", freq="h", name="datetime", tz="UTC")
        self.raw = pd.DataFrame({
            "TT_TU": np.arange(len(index)) * .01,
            "RF_TU": 70., "R1": .1, "SD_SO": 20., "F": 2., "P0": 980.,
        }, index=index)
        self.cfg = PrecipitationConfig(lookback_hours=4,
                                       train_end_exclusive="2020-01-01",
                                       validation_end_exclusive="2020-01-03",
                                       accepted_target_quality_levels=None)

    def observations(self):
        self.raw.to_csv(self.source)
        return read_observations(self.source)

    def test_missing_and_invalid_are_preserved_separately(self):
        self.raw.iloc[5, self.raw.columns.get_loc("R1")] = np.nan
        self.raw.iloc[6, self.raw.columns.get_loc("R1")] = -999
        self.raw.iloc[7, self.raw.columns.get_loc("R1")] = -1
        self.raw.iloc[8, self.raw.columns.get_loc("R1")] = .15
        obs = self.observations()
        self.assertTrue(obs.R1.iloc[5:9].isna().all())
        self.assertTrue(obs.R1_source_missing.iloc[5:7].all())
        self.assertTrue(obs.R1_invalid.iloc[7:9].all())
        self.assertFalse(obs.R1_observed.iloc[5:9].any())

    def test_missing_timestamp_becomes_explicit_missing_row(self):
        removed = self.raw.index[30]
        self.raw = self.raw.drop(removed)
        obs = self.observations()
        self.assertFalse(obs.loc[removed, "source_row_present"])
        self.assertTrue(obs.loc[removed, WEATHER_COLUMNS].isna().all())
        self.assertTrue(obs.loc[removed, [c + "_source_missing" for c in WEATHER_COLUMNS]].all())

    def test_duplicate_and_non_hourly_timestamps_are_rejected(self):
        self.raw = pd.concat([self.raw, self.raw.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.observations()
        self.raw = self.raw.iloc[:-1]
        self.raw.index = self.raw.index + pd.Timedelta(minutes=10)
        with self.assertRaisesRegex(ValueError, "hourly grid"):
            self.observations()

    def test_targets_use_exact_future_hours_and_tenth_mm_boundary(self):
        issue = pd.Timestamp("2019-12-30 12:00", tz="UTC")
        self.raw.loc[issue + pd.Timedelta(hours=1):issue + pd.Timedelta(hours=3), "R1"] = [2.3, .4, .3]
        self.raw.loc[issue + pd.Timedelta(hours=4):issue + pd.Timedelta(hours=6), "R1"] = [0., 0., 0.]
        samples = build_samples(self.observations(), self.cfg).set_index(["issue_reference_time", "horizon_h"])
        near = samples.loc[(issue, 3)]
        far = samples.loc[(issue, 6)]
        self.assertEqual(near.target_mm, 3.)
        self.assertEqual(near.target_class, 2)
        self.assertEqual(near.target_rain, 1)
        self.assertEqual(far.target_mm, 0.)
        self.assertEqual(far.target_class, 0)
        self.assertEqual(near.input_end, issue - pd.Timedelta(hours=1))
        self.assertEqual(near.input_start, issue - pd.Timedelta(hours=4))
        self.assertEqual(near.issue_time, issue - pd.Timedelta(minutes=10))
        self.assertEqual(near.target_end_inclusive - near.issue_time, pd.Timedelta(hours=3))
        self.assertEqual(near.target_start_exclusive, near.issue_time)
        self.assertEqual(near.target_reference_end, issue + pd.Timedelta(hours=3))
        self.assertEqual(near.assumed_latest_input_available_at, near.issue_time)

    def test_missing_target_is_never_filled_or_called_dry(self):
        issue = pd.Timestamp("2019-12-30 12:00", tz="UTC")
        self.raw.loc[issue + pd.Timedelta(hours=2), "R1"] = np.nan
        samples = build_samples(self.observations(), self.cfg)
        row = samples.loc[samples.issue_reference_time.eq(issue) & samples.horizon_h.eq(3)].iloc[0]
        self.assertEqual(row.target_observed_hours, 2)
        self.assertTrue(pd.isna(row.target_mm))
        self.assertTrue(pd.isna(row.target_class))
        self.assertTrue(pd.isna(row.target_rain))
        self.assertFalse(row.eligible)
        self.assertIn("incomplete_target", row.exclusion_reason)

    def test_split_crossing_targets_are_excluded(self):
        samples = build_samples(self.observations(), self.cfg)
        crossing = samples.loc[samples.issue_reference_time.eq("2019-12-31 21:00") & samples.horizon_h.eq(6)].iloc[0]
        self.assertTrue(crossing.target_complete)
        self.assertFalse(crossing.within_split)
        self.assertFalse(crossing.eligible)
        self.assertIn("split_boundary", crossing.exclusion_reason)
        first_val = samples.loc[samples.issue_reference_time.eq("2020-01-01 03:00") & samples.horizon_h.eq(3)].iloc[0]
        self.assertEqual(first_val.split, "validation")
        self.assertTrue(first_val.eligible)
        self.assertLess(first_val.input_start, pd.Timestamp(self.cfg.train_end_exclusive, tz="UTC"))
        # A 00:00 reference target ends at 23:50 of the previous UTC day.
        previous = samples.loc[samples.issue_reference_time.eq("2019-12-31 21:00") & samples.horizon_h.eq(3)].iloc[0]
        self.assertTrue(previous.within_split)
        self.assertEqual(previous.target_end_inclusive, pd.Timestamp("2019-12-31 23:50", tz="UTC"))

    def test_incomplete_history_and_end_of_file_are_audited(self):
        samples = build_samples(self.observations(), self.cfg)
        first = samples.iloc[0]
        last = samples.iloc[-1]
        self.assertFalse(first.history_complete)
        self.assertFalse(first.eligible)
        self.assertFalse(last.target_complete)
        self.assertFalse(last.eligible)

    def test_no_future_mutation_changes_past_inputs_or_training_statistics(self):
        obs = self.observations()
        obs.loc["2019-12-30 10:00":"2019-12-30 15:00", "TT_TU"] = np.nan
        original, state = prepare_features(obs, self.cfg)
        changed = obs.copy()
        future = pd.Timestamp("2020-01-02 12:00", tz="UTC")
        changed.loc[future:, WEATHER_COLUMNS] += 100
        transformed, changed_state = prepare_features(changed, self.cfg)
        self.assertEqual(state, changed_state)
        pd.testing.assert_frame_equal(original.loc[original.index < future],
                                      transformed.loc[transformed.index < future])
        training = obs.loc[obs.index < pd.Timestamp(self.cfg.train_end_exclusive, tz="UTC"), "TT_TU"]
        self.assertAlmostEqual(state["statistics"]["TT_TU"]["mean"], training.mean())

    def test_carry_is_limited_and_rain_and_sun_are_not_carried(self):
        start = pd.Timestamp("2019-12-30 10:00", tz="UTC")
        self.raw.loc[start:start + pd.Timedelta(hours=4), "TT_TU"] = np.nan
        self.raw.loc[start - pd.Timedelta(hours=1), "R1"] = 2.
        self.raw.loc[start, ["R1", "SD_SO"]] = np.nan
        features, state = prepare_features(self.observations(), self.cfg)
        self.assertEqual(features.loc[start:start + pd.Timedelta(hours=4), "TT_TU_carried"].tolist(), [1., 1., 1., 0., 0.])
        for col in ["R1", "SD_SO"]:
            stats = state["statistics"][col]
            restored = features.loc[start, col] * stats["std"] + stats["mean"]
            self.assertAlmostEqual(restored, stats["median"], places=6)
            self.assertEqual(features.loc[start, col + "_missing"], 1.)

    def test_models_receive_identical_windows_and_export_roundtrips(self):
        self.observations()
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        output = Path(self.temp.name) / "built"
        summary, manifest = build_dataset(self.source, output, self.cfg)
        bundle = load_precipitation_bundle(output)
        x, y, rows = bundle.arrays(3, "validation")
        flat, flat_y, flat_rows = bundle.arrays(3, "validation", flatten=True)
        np.testing.assert_array_equal(x.reshape(len(x), -1), flat)
        np.testing.assert_array_equal(y, flat_y)
        pd.testing.assert_frame_equal(rows, flat_rows)
        self.assertEqual(x.shape[1:], (4, 20))
        first = rows.iloc[0]
        expected = bundle.features.loc[first.input_start:first.input_end].to_numpy()
        np.testing.assert_array_equal(x[0], expected)
        self.assertEqual(manifest["source_sha256"], before)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
        self.assertEqual(summary.eligible.sum(), bundle.samples.eligible.sum())

    def test_invalid_configuration_and_empty_training_feature_are_rejected(self):
        with self.assertRaises(ValueError):
            PrecipitationConfig(horizons_hours=(2,)).validate()
        self.raw.loc[self.raw.index < pd.Timestamp(self.cfg.train_end_exclusive, tz="UTC"), "RF_TU"] = np.nan
        with self.assertRaisesRegex(ValueError, "observed training"):
            prepare_features(self.observations(), self.cfg)

    def test_utc_survives_parquet_and_local_dst_conversion(self):
        for text in ["2024-03-31 00:00", "2024-10-27 00:00"]:
            times = pd.date_range(text, periods=5, freq="h", tz="UTC")
            local = times.tz_convert("Europe/Berlin")
            pd.testing.assert_index_equal(local.tz_convert("UTC"), times)
            self.assertFalse(local.duplicated().any())
        # Naive CSV hours are localized to the verified UTC source zone.
        naive = self.raw.copy()
        naive.index = naive.index.tz_localize(None)
        naive.to_csv(self.source)
        obs = read_observations(self.source)
        self.assertEqual(str(obs.index.tz), "UTC")

    def test_formally_checked_targets_remain_observed_but_are_excluded(self):
        from dataclasses import replace
        obs = self.observations()
        obs["R1_QN"] = 3
        reference = pd.Timestamp("2019-12-30 12:00", tz="UTC")
        obs.loc[reference + pd.Timedelta(hours=2), "R1_QN"] = 1
        config = replace(self.cfg, accepted_target_quality_levels=(3, 10))
        samples = build_samples(obs, config)
        row = samples.loc[samples.issue_reference_time.eq(reference) & samples.horizon_h.eq(3)].iloc[0]
        self.assertTrue(row.target_complete)
        self.assertFalse(row.target_quality_acceptable)
        self.assertFalse(row.eligible)
        self.assertEqual(row.target_quality_known_hours, 3)
        self.assertEqual(row.target_class, 1)
        self.assertIn("target_quality", row.exclusion_reason)

    def test_wrong_csv_metadata_binding_is_rejected(self):
        import json
        self.observations()
        evidence = Path(self.temp.name) / "audit.json"
        evidence.write_text(json.dumps({"timezone_verified": True, "timezone": "UTC", "source_csv_sha256": "wrong"}))
        with self.assertRaisesRegex(ValueError, "exact source CSV"):
            build_dataset(self.source, Path(self.temp.name) / "bad", self.cfg, metadata_audit=evidence)


if __name__ == "__main__":
    unittest.main()
