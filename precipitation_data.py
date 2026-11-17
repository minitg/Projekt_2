"""Auditable hourly inputs and observed precipitation targets for model comparison.

Original station metadata establishes UTC for the pilot period. Rain interval
ends are estimated at reference time minus 10 minutes from a primary-data audit;
real publication times are not available in the historical archive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = ["TT_TU", "RF_TU", "R1", "SD_SO", "F", "P0"]
CONTINUOUS_COLUMNS = ["TT_TU", "RF_TU", "F", "P0"]


@dataclass(frozen=True)
class PrecipitationConfig:
    lookback_hours: int = 48
    horizons_hours: tuple[int, ...] = (3, 6, 12, 24)
    observation_delay_hours: int = 1
    forward_fill_limit_hours: int = 3
    train_end_exclusive: str = "2022-01-01"
    validation_end_exclusive: str = "2024-01-01"
    upper_class_threshold_mm: float = 3.0
    source_start: str | None = "2018-01-01"
    reference_offset_minutes: int = -10
    accepted_target_quality_levels: tuple[int, ...] | None = (3, 10)

    def validate(self) -> None:
        if self.lookback_hours < 1 or self.observation_delay_hours < 0:
            raise ValueError("lookback must be positive and observation delay nonnegative")
        if self.forward_fill_limit_hours < 0:
            raise ValueError("forward-fill limit must be nonnegative")
        if (not self.horizons_hours or len(set(self.horizons_hours)) != len(self.horizons_hours)
                or any(h < 3 or h % 3 for h in self.horizons_hours)):
            raise ValueError("horizons must be unique positive multiples of 3 hours")
        train_end = utc_timestamp(self.train_end_exclusive)
        val_end = utc_timestamp(self.validation_end_exclusive)
        if train_end >= val_end:
            raise ValueError("split boundaries must be ordered UTC timestamps")
        if self.source_start is not None and utc_timestamp(self.source_start) >= train_end:
            raise ValueError("source start must precede the training cutoff")
        if not -59 <= self.reference_offset_minutes <= 0:
            raise ValueError("reference offset must be between -59 and 0 minutes")
        if self.accepted_target_quality_levels is not None and (
                not self.accepted_target_quality_levels
                or not set(self.accepted_target_quality_levels).issubset({1, 2, 3, 5, 7, 8, 9, 10})):
            raise ValueError("choose explicit DWD quality procedure codes")
        if (not np.isfinite(self.upper_class_threshold_mm)
                or self.upper_class_threshold_mm <= 0
                or not np.isclose(self.upper_class_threshold_mm * 10,
                                   round(self.upper_class_threshold_mm * 10))):
            raise ValueError("class threshold must be positive and in tenths of a millimeter")


def utc_timestamp(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")


def read_observations(path: str | Path) -> pd.DataFrame:
    """Keep available source masks; reject ambiguous timestamps or bad schema.

    Range checks are deliberately broad and are not station-specific QC.
    'observed' means finite / plausible in this CSV, not verified raw-sensor QC.
    """
    source = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if source.empty:
        raise ValueError("hourly source is empty")
    if not isinstance(source.index, pd.DatetimeIndex) or source.index.hasnans:
        raise ValueError("source timestamps must be valid datetimes")
    # For this source, the default CLI verifies the UTC declaration against
    # the original metadata and the complete CSV hash before reading it.
    source.index = (source.index.tz_localize("UTC") if source.index.tz is None
                    else source.index.tz_convert("UTC"))
    if source.index.duplicated().any():
        raise ValueError("duplicate source timestamps must be resolved explicitly")
    if not source.index.equals(source.index.floor("h")):
        raise ValueError("source timestamps must be on the hourly grid")
    absent = set(WEATHER_COLUMNS).difference(source.columns)
    if absent:
        raise ValueError(f"missing source columns: {sorted(absent)}")

    index = pd.date_range(source.index.min(), source.index.max(), freq="h", name="datetime")
    values = source[WEATHER_COLUMNS].apply(pd.to_numeric, errors="raise").reindex(index)
    missing = values.isna() | values.eq(-999)
    invalid = ~np.isfinite(values) & ~missing
    bounds = {
        "TT_TU": values.TT_TU.between(-90, 60),
        "RF_TU": values.RF_TU.between(0, 100),
        "R1": values.R1.ge(0) & np.isclose(values.R1 * 10, (values.R1 * 10).round(), atol=1e-6, rtol=0),
        "SD_SO": values.SD_SO.between(0, 60),
        "F": values.F.ge(0),
        "P0": values.P0.gt(0),
    }
    for col, plausible in bounds.items():
        invalid[col] |= ~plausible & ~missing[col]
    observed = ~(missing | invalid)
    cleaned = values.where(observed)
    result = cleaned.copy()
    for col in WEATHER_COLUMNS:
        result[f"{col}_observed"] = observed[col]
        result[f"{col}_source_missing"] = missing[col]
        result[f"{col}_invalid"] = invalid[col]
    result["source_row_present"] = result.index.isin(source.index)
    # Wind direction remains in the source CSV; it is not silently interpreted
    # as a linear feature or north when the direction is undefined.
    return result


def prepare_features(observations: pd.DataFrame, cfg: PrecipitationConfig):
    cfg.validate()
    raw = observations[WEATHER_COLUMNS]
    training = raw.loc[raw.index < utc_timestamp(cfg.train_end_exclusive)]
    if training.empty or training.notna().sum().eq(0).any():
        raise ValueError("each physical feature needs at least one observed training value")
    median = training.median()
    mean = training.mean()
    std = training.std(ddof=0).fillna(0).replace(0, 1)

    filled = raw.copy()
    carried = pd.DataFrame(False, index=raw.index, columns=CONTINUOUS_COLUMNS)
    if cfg.forward_fill_limit_hours:
        filled[CONTINUOUS_COLUMNS] = raw[CONTINUOUS_COLUMNS].ffill(limit=cfg.forward_fill_limit_hours)
        carried = raw[CONTINUOUS_COLUMNS].isna() & filled[CONTINUOUS_COLUMNS].notna()
    # Never forward-fill precipitation or sunshine sums. Missing input values
    # may use a training median, but the missingness flags always remain set.
    filled = filled.fillna(median)
    features = ((filled - mean) / std).astype(np.float32)
    for col in WEATHER_COLUMNS:
        features[f"{col}_missing"] = raw[col].isna().astype(np.float32)
    for col in CONTINUOUS_COLUMNS:
        features[f"{col}_carried"] = carried[col].astype(np.float32)
    features["hour_sin"] = np.sin(2 * np.pi * raw.index.hour / 24).astype(np.float32)
    features["hour_cos"] = np.cos(2 * np.pi * raw.index.hour / 24).astype(np.float32)
    features["day_sin"] = np.sin(2 * np.pi * (raw.index.dayofyear - 1) / 365.25).astype(np.float32)
    features["day_cos"] = np.cos(2 * np.pi * (raw.index.dayofyear - 1) / 365.25).astype(np.float32)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("nonfinite model input after preprocessing")
    state = {
        "fit_end_exclusive": cfg.train_end_exclusive,
        "fit_start": str(training.index.min()),
        "statistics_source": "observed, plausible training values only; before imputation",
        "time_zone": "UTC",
        "statistics": {col: {"median": float(median[col]), "mean": float(mean[col]),
                             "std": float(std[col]), "observed_count": int(training[col].notna().sum())}
                       for col in WEATHER_COLUMNS},
        "forward_fill_columns": CONTINUOUS_COLUMNS,
        "forward_fill_limit_hours": cfg.forward_fill_limit_hours,
        "feature_columns": features.columns.tolist(),
        "config": asdict(cfg),
    }
    return features, state


def _split(times, cfg):
    return np.where(times < utc_timestamp(cfg.train_end_exclusive), "train",
                    np.where(times < utc_timestamp(cfg.validation_end_exclusive), "validation", "evaluation"))


def build_samples(observations: pd.DataFrame, cfg: PrecipitationConfig) -> pd.DataFrame:
    cfg.validate()
    first, last = observations.index.min(), observations.index.max()
    references = pd.date_range(first.ceil("3h"), last.floor("3h"), freq="3h")
    offset = pd.to_timedelta(cfg.reference_offset_minutes, unit="min")
    issues = references + offset
    if references.empty:
        raise ValueError("source has no issue times on a 3-hour grid")
    rain = observations.R1
    if cfg.accepted_target_quality_levels is not None and "R1_QN" not in observations:
        raise ValueError("target quality policy requires original R1_QN values from the source audit")
    # Input start/end are DWD reference timestamps. Actual issuance is on the
    # :50 schedule; the assumed delay is relative to the estimated observation
    # interval end, not a promise that CDC has already published the record.
    input_end = references - pd.to_timedelta(cfg.observation_delay_hours, unit="h")
    input_start = input_end - pd.to_timedelta(cfg.lookback_hours - 1, unit="h")
    missing_per_hour = observations[WEATHER_COLUMNS].isna().sum(axis=1)
    missing_in_window = missing_per_hour.rolling(cfg.lookback_hours, min_periods=cfg.lookback_hours).sum()
    output = []
    for horizon in cfg.horizons_hours:
        target_reference_end = references + pd.to_timedelta(horizon, unit="h")
        target_end = target_reference_end + offset
        target_start = target_end - pd.Timedelta(hours=3)
        measurements = np.column_stack([
            rain.reindex(target_reference_end - pd.to_timedelta(hours, unit="h")).to_numpy()
            for hours in [2, 1, 0]
        ])
        observed_count = np.isfinite(measurements).sum(axis=1)
        target_complete = observed_count == 3
        qn = np.column_stack([
            observations.R1_QN.reindex(target_reference_end - pd.to_timedelta(hours, unit="h")).to_numpy(dtype=float, na_value=np.nan)
            for hours in [2, 1, 0]
        ]) if "R1_QN" in observations else np.full(measurements.shape, np.nan)
        quality_acceptable = (np.isin(qn, cfg.accepted_target_quality_levels).all(axis=1)
                              if cfg.accepted_target_quality_levels is not None else np.ones(len(issues), dtype=bool))
        amount = np.where(target_complete, np.nansum(measurements, axis=1).round(1), np.nan)
        classes = pd.array(np.where(np.isnan(amount), np.nan,
                          np.where(amount == 0, 0, np.where(amount < cfg.upper_class_threshold_mm, 1, 2))), dtype="Int8")
        binary = pd.array(np.where(np.isnan(amount), np.nan, amount > 0), dtype="Int8")
        split = _split(issues, cfg)
        # Partition using the estimated physical interval, including its start.
        # Raw target reference stamps are retained separately for auditability.
        within_split = ((split == _split(target_end, cfg))
                        & (split == _split(target_start, cfg)))
        full_history = (input_start >= first) & (input_end <= last)
        eligible = full_history & target_complete & within_split & quality_acceptable
        reason = np.full(len(issues), "", dtype=object)
        for failed, text in [(~full_history, "insufficient_history"),
                             (~target_complete, "incomplete_target"),
                             (~quality_acceptable, "target_quality"),
                             (~within_split, "split_boundary")]:
            reason[failed] = [f"{old};{text}".strip(";") for old in reason[failed]]
        frame = pd.DataFrame({
            "issue_time": issues, "issue_reference_time": references, "horizon_h": horizon,
            "input_start": input_start, "input_end": input_end,
            "input_end_index": observations.index.get_indexer(input_end),
            "input_missing_values": pd.array(missing_in_window.reindex(input_end).to_numpy(), dtype="Int16"),
            "target_start_exclusive": target_start, "target_end_inclusive": target_end,
            "target_reference_end": target_reference_end,
            "input_rain_interval_end_estimate": input_end + offset,
            "assumed_latest_input_available_at": input_end + offset + pd.to_timedelta(cfg.observation_delay_hours, unit="h"),
            "target_observed_hours": observed_count,
            "target_quality_known_hours": np.isfinite(qn).sum(axis=1),
            "target_quality_acceptable": quality_acceptable,
            "target_mm": amount, "target_rain": binary, "target_class": classes,
            "split": split, "history_complete": full_history,
            "target_complete": target_complete, "within_split": within_split,
            "eligible": eligible, "exclusion_reason": reason,
        })
        output.append(frame)
    return pd.concat(output, ignore_index=True).sort_values(["issue_time", "horizon_h"]).reset_index(drop=True)


@dataclass
class PrecipitationBundle:
    features: pd.DataFrame
    samples: pd.DataFrame
    config: PrecipitationConfig

    def select(self, horizon_h: int, split: str) -> pd.DataFrame:
        if horizon_h not in self.config.horizons_hours or split not in {"train", "validation", "evaluation"}:
            raise ValueError("unknown horizon or split")
        return self.samples.loc[self.samples.eligible & self.samples.horizon_h.eq(horizon_h)
                                & self.samples.split.eq(split)].copy()

    def arrays(self, horizon_h: int, split: str, *, flatten: bool = False,
               binary: bool = False):
        """Same information for LSTM (N,48,F) and XGBoost (N,48*F).

        Samples retain their source missingness; all numerical fills and scales
        are from preprocessing.json. Returned arrays contain eligible rows only.
        """
        samples = self.select(horizon_h, split)
        lookback = self.config.lookback_hours
        values = self.features.to_numpy(dtype=np.float32)
        if len(values) < lookback:
            raise ValueError("not enough source history for one window")
        windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)
        starts = samples.input_end_index.to_numpy(dtype=int) - lookback + 1
        x = windows[starts]
        if flatten:
            x = x.reshape(len(x), lookback * values.shape[1])
        y = samples["target_rain" if binary else "target_class"].to_numpy(dtype=np.int64)
        return x, y, samples.reset_index(drop=True)


def load_precipitation_bundle(directory: str | Path) -> PrecipitationBundle:
    import json
    directory = Path(directory)
    state = json.loads((directory / "preprocessing.json").read_text())
    config_values = state["config"]
    # Old v1 artifacts keep their original hourly reference convention.
    config_values.setdefault("source_start", None)
    config_values.setdefault("reference_offset_minutes", 0)
    config_values.setdefault("accepted_target_quality_levels", None)
    if config_values["accepted_target_quality_levels"] is not None:
        config_values["accepted_target_quality_levels"] = tuple(config_values["accepted_target_quality_levels"])
    config_values["horizons_hours"] = tuple(config_values["horizons_hours"])
    cfg = PrecipitationConfig(**config_values)
    cfg.validate()
    features = pd.read_parquet(directory / "hourly_features.parquet")
    samples = pd.read_parquet(directory / "samples.parquet")
    if features.columns.tolist() != state["feature_columns"]:
        raise ValueError("saved feature order does not match preprocessing state")
    return PrecipitationBundle(features, samples, cfg)
