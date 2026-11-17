"""Three-class precipitation inputs, with labels kept outside prediction records.

This is a model-independent data contract, not a Chronos/Timestral classifier.
Numerical forecasters still need a classification adapter or a validated
distribution-to-class conversion. Class IDs must never be forecast as amounts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from precipitation_data import (
    CONTINUOUS_COLUMNS, WEATHER_COLUMNS, PrecipitationConfig,
    load_precipitation_bundle,
)


CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
CASE_COLUMNS = [
    "issue_time", "issue_reference_time", "horizon_h", "split",
    "input_start", "input_end", "input_end_index",
    "input_rain_interval_end_estimate", "assumed_latest_input_available_at",
    "target_start_exclusive", "target_end_inclusive", "target_reference_end",
]
DATA_FILES = [
    "cases.parquet", "labels.parquet", "hourly_features.parquet",
    "hourly_context.parquet", "schema.json", "preprocessing.json",
    "class_balance.json",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Only deterministic UTC calendar information; no future measurements."""
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * times.hour / 24),
        "hour_cos": np.cos(2 * np.pi * times.hour / 24),
        "day_sin": np.sin(2 * np.pi * (times.dayofyear - 1) / 365.25),
        "day_cos": np.cos(2 * np.pi * (times.dayofyear - 1) / 365.25),
    }, index=times).astype(np.float32)


def class_schema(threshold_mm: float) -> dict:
    return {
        "task": "multiclass_classification", "primary_target": "target_class",
        "class_ids": [0, 1, 2], "probability_column_order": ["p_class0", "p_class1", "p_class2"],
        "target_duration_hours": 3, "upper_class_threshold_mm": threshold_mm,
        "classes": [
            {"id": 0, "name": "dry", "description": "0 mm / 3h", "exact_mm": 0.0},
            {"id": 1, "name": "wet_below_threshold",
             "description": f">0 bis <{threshold_mm:g} mm / 3h",
             "exclusive_min_mm": 0.0, "exclusive_max_mm": threshold_mm},
            {"id": 2, "name": "wet_at_or_above_threshold",
             "description": f">={threshold_mm:g} mm / 3h", "inclusive_min_mm": threshold_mm},
        ],
        "class_ids_are": "categorical labels; not rainfall amounts or an ordinal regression target",
        "warning_thresholds": False,
        "input_roles": {
            "sequence": "hourly_features.parquet: same scaled 20-feature history as parent v2",
            "historical_forecast_target": "hourly_context.parquet: R1_observed_mm; NaN stays unknown",
            "past_covariates": "original-unit physical inputs, missing/carried flags and calendar; R1 supplied as target history",
            "known_future_covariates": CALENDAR_COLUMNS,
            "labels_and_diagnostics": "labels.parquet; never part of prediction records",
        },
        "forecast_adapter": {
            "grid": "hourly DWD reference timestamps; physical rain interval ends are estimated at reference + offset",
            "prediction_length": "horizon_h + observation_delay_hours; includes withheld observation gap",
            "class_target": "sum of only the last three forecast hours, not all hours up to the horizon",
            "probabilities": "require a trained classification head/adapter or a justified predictive distribution; do not threshold a mean to obtain probabilities",
            "joint_samples": "sum each joint three-hour sample path before counting class frequencies; marginal quantiles cannot simply be added",
        },
    }


def build_classification_dataset(source_dir: str | Path, output_dir: str | Path) -> dict:
    """Add a self-contained classification view without changing the parent v2."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("classification output must be separate from the parent dataset")
    parent = json.loads((source_dir / "manifest.json").read_text())
    if parent.get("dataset_version") != "precipitation_v2":
        raise ValueError("build the audited precipitation_v2 parent first")
    bundle = load_precipitation_bundle(source_dir)
    cfg = bundle.config
    preprocessing = json.loads((source_dir / "preprocessing.json").read_text())
    if json.loads(json.dumps(asdict(cfg))) != parent["config"]:
        raise ValueError("parent manifest and preprocessing configuration disagree")
    observations = pd.read_parquet(source_dir / "hourly_observations.parquet")
    if (not observations.index.equals(bundle.features.index)
            or observations.index.tz is None or str(observations.index.tz) != "UTC"
            or observations.index.has_duplicates
            or not observations.index.equals(pd.date_range(
                observations.index.min(), observations.index.max(), freq="h", name=observations.index.name))):
        raise ValueError("parent inputs must share a complete, unique UTC hourly grid")

    selected = bundle.samples.loc[bundle.samples.eligible].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("parent has no eligible classification cases")
    # Recheck labels against the actual three observations, not an imputed input.
    measured = np.column_stack([
        observations.R1.reindex(selected.target_reference_end - pd.Timedelta(hours=lag)).to_numpy()
        for lag in (2, 1, 0)
    ])
    if not np.isfinite(measured).all() or (measured < 0).any():
        raise ValueError("eligible labels require three nonnegative observed rain hours")
    sums = measured.sum(axis=1).round(1)
    expected = np.where(sums == 0, 0, np.where(sums < cfg.upper_class_threshold_mm, 1, 2))
    if (not np.array_equal(expected, selected.target_class.to_numpy())
            or not np.allclose(sums, selected.target_mm.to_numpy())):
        raise ValueError("parent class labels disagree with the observed three-hour sums")
    if cfg.accepted_target_quality_levels is not None:
        quality = np.column_stack([
            observations.R1_QN.reindex(selected.target_reference_end - pd.Timedelta(hours=lag)).to_numpy(dtype=float, na_value=np.nan)
            for lag in (2, 1, 0)
        ])
        if not np.isin(quality, cfg.accepted_target_quality_levels).all():
            raise ValueError("parent eligible cases violate the target quality policy")

    case_ids = (str(parent["station_id"]) + "_"
                + selected.issue_reference_time.dt.strftime("%Y%m%dT%H%MZ")
                + "_h" + selected.horizon_h.astype(str).str.zfill(2))
    if case_ids.duplicated().any():
        raise ValueError("parent contains duplicate issue/horizon cases")
    cases = selected[CASE_COLUMNS].copy()
    cases.insert(0, "case_id", case_ids)
    labels = pd.DataFrame({
        "case_id": case_ids, "target_class": selected.target_class.astype(np.int8),
        "observed_target_mm": sums,
    })
    # Rebuild physical inputs directly from observations, avoiding round-off
    # from inverse scaling. Use exactly the parent's causal filling policy.
    physical = observations[WEATHER_COLUMNS].copy()
    if cfg.forward_fill_limit_hours:
        physical[CONTINUOUS_COLUMNS] = physical[CONTINUOUS_COLUMNS].ffill(limit=cfg.forward_fill_limit_hours)
    for col in WEATHER_COLUMNS:
        stats = preprocessing["statistics"][col]
        physical[col] = physical[col].fillna(stats["median"])
        scaled = ((physical[col] - stats["mean"]) / stats["std"]).astype(np.float32)
        if not np.array_equal(scaled.to_numpy(), bundle.features[col].to_numpy()):
            raise ValueError(f"parent preprocessing is not reproducible for {col}")
    context = bundle.features.copy()
    context[WEATHER_COLUMNS] = physical
    context["R1_observed_mm"] = observations.R1
    if not np.array_equal(calendar_features(context.index).to_numpy(), context[CALENDAR_COLUMNS].to_numpy()):
        raise ValueError("parent calendar feature convention is inconsistent")

    balance = []
    for (horizon, split), group in selected.groupby(["horizon_h", "split"], sort=True):
        counts = group.target_class.value_counts()
        balance.append({
            "horizon_h": int(horizon), "split": split, "n": len(group),
            "class_counts": {str(k): int(counts.get(k, 0)) for k in (0, 1, 2)},
            "class_frequencies": {str(k): float(counts.get(k, 0) / len(group)) for k in (0, 1, 2)},
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(output_dir / "cases.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    bundle.features.to_parquet(output_dir / "hourly_features.parquet")
    context.to_parquet(output_dir / "hourly_context.parquet")
    for name, content in [
        ("schema.json", class_schema(cfg.upper_class_threshold_mm)),
        ("preprocessing.json", preprocessing), ("class_balance.json", balance),
    ]:
        (output_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    source_names = ["manifest.json", "preprocessing.json", "hourly_features.parquet",
                    "hourly_observations.parquet", "samples.parquet"]
    manifest = {
        "dataset_version": "precipitation_classification_v1", "task": "multiclass_classification",
        "parent_dataset_version": parent["dataset_version"],
        "parent_directory": str(source_dir.resolve()),
        "parent_files_sha256": {name: file_sha256(source_dir / name) for name in source_names},
        "station_id": parent["station_id"], "source_sha256": parent["source_sha256"],
        "config": asdict(cfg), "cases": len(cases), "feature_count": len(bundle.features.columns),
        "hourly_rows": len(context), "resampling": "none; identical eligible cases to parent v2",
        "timezone_verified": parent["timezone_verified"],
        "interval_end_evidence": parent["interval_end_evidence"],
        "availability_verified": parent["availability_verified"],
        "operational_ready": parent["operational_ready"],
        "evaluation_status": parent["evaluation_status"],
        "files_sha256": {name: file_sha256(output_dir / name) for name in DATA_FILES},
        "code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                        ["classification_data.py", "build_precipitation_classification_dataset.py"]},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


@dataclass
class PrecipitationClassificationBundle:
    features: pd.DataFrame
    context: pd.DataFrame
    cases: pd.DataFrame
    labels: pd.DataFrame
    config: PrecipitationConfig

    def select(self, horizon_h: int, split: str) -> pd.DataFrame:
        if horizon_h not in self.config.horizons_hours or split not in {"train", "validation", "evaluation"}:
            raise ValueError("unknown horizon or split")
        return self.cases.loc[self.cases.horizon_h.eq(horizon_h) & self.cases.split.eq(split)].reset_index(drop=True)

    def arrays(self, horizon_h: int, split: str, *, flatten: bool = False):
        """Return X, categorical y and case metadata; y is never an input."""
        cases = self.select(horizon_h, split)
        values = self.features.to_numpy(dtype=np.float32)
        lookback = self.config.lookback_hours
        windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)
        starts = cases.input_end_index.to_numpy(dtype=int) - lookback + 1
        x = windows[starts]
        if flatten:
            x = x.reshape(len(cases), lookback * values.shape[1])
        y = self.labels.set_index("case_id").loc[cases.case_id, "target_class"].to_numpy(dtype=np.int64)
        return x, y, cases

    def foundation_records(self, horizon_h: int, split: str) -> Iterator[dict]:
        """Model-neutral hourly records in original units, with no future weather.

        These are adapter inputs, not a native API call for any one library.
        The iterator deliberately does not expose labels or observed target sums.
        """
        for row in self.select(horizon_h, split).itertuples(index=False):
            start = row.input_end_index - self.config.lookback_hours + 1
            history = self.context.iloc[start:row.input_end_index + 1]
            length = horizon_h + self.config.observation_delay_hours
            future_times = pd.date_range(row.input_end + pd.Timedelta(hours=1), periods=length, freq="h")
            future_calendar = calendar_features(future_times)
            yield {
                "case_id": row.case_id,
                "context_timestamps": history.index.copy(),
                "target": history.R1_observed_mm.to_numpy(dtype=float, copy=True),
                "past_covariates": {col: history[col].to_numpy(copy=True) for col in self.features.columns if col != "R1"},
                "future_covariates": {col: future_calendar[col].to_numpy(copy=True) for col in CALENDAR_COLUMNS},
                "future_timestamps": future_times,
                "prediction_length": length,
                "class_target_hour_indices": np.arange(length - 3, length),
            }


def load_classification_bundle(directory: str | Path) -> PrecipitationClassificationBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset_version") != "precipitation_classification_v1":
        raise ValueError("unsupported classification dataset version")
    for name in DATA_FILES:
        if file_sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"classification dataset checksum mismatch: {name}")
    state = json.loads((directory / "preprocessing.json").read_text())
    config = state["config"]
    config["horizons_hours"] = tuple(config["horizons_hours"])
    if config["accepted_target_quality_levels"] is not None:
        config["accepted_target_quality_levels"] = tuple(config["accepted_target_quality_levels"])
    cfg = PrecipitationConfig(**config)
    cfg.validate()
    features = pd.read_parquet(directory / "hourly_features.parquet")
    context = pd.read_parquet(directory / "hourly_context.parquet")
    cases = pd.read_parquet(directory / "cases.parquet")
    labels = pd.read_parquet(directory / "labels.parquet")
    if (features.columns.tolist() != state["feature_columns"]
            or not features.index.equals(context.index)
            or cases.case_id.duplicated().any() or labels.case_id.duplicated().any()
            or not cases.case_id.equals(labels.case_id)
            or not labels.target_class.isin([0, 1, 2]).all()):
        raise ValueError("inconsistent classification feature, case or label schema")
    return PrecipitationClassificationBundle(features, context, cases, labels, cfg)
