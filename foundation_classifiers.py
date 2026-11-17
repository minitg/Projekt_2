"""Frozen Chronos-2/Timestral forecasts followed by a supervised three-class head.

The classifier learns from the nine marginal quantiles of each of the three
target hours. Those quantiles are features, never an asserted joint CDF of the
three-hour sum. Foundation weights remain frozen; the full system is supervised.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification_data import CALENDAR_COLUMNS


QUANTILE_LEVELS = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
FEATURE_NAMES = [f"target_hour_{hour}_q{q:g}_mm" for hour in (1, 2, 3) for q in QUANTILE_LEVELS]


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_records(records: list[dict]) -> int:
    if not records:
        raise ValueError("forecast batch must not be empty")
    prediction_length = records[0]["prediction_length"]
    past_keys = set(records[0]["past_covariates"])
    for record in records:
        if "target_class" in record or "observed_target_mm" in record:
            raise ValueError("classification labels must never enter the forecaster")
        if record["prediction_length"] != prediction_length or prediction_length < 3:
            raise ValueError("one batch must share a prediction length of at least three")
        if (set(record["future_covariates"]) != set(CALENDAR_COLUMNS)
                or not set(CALENDAR_COLUMNS).issubset(past_keys)
                or set(record["past_covariates"]) != past_keys):
            raise ValueError("only shared UTC calendar covariates may be known into the future")
        history = np.asarray(record["target"])
        if history.ndim != 1 or not len(history) or np.isinf(history).any():
            raise ValueError("expected one historical rain series with NaN for missing values")
        if len(record["context_timestamps"]) != len(history):
            raise ValueError("historical values and timestamps must have equal lengths")
        future_times = record["future_timestamps"]
        expected_times = pd.date_range(record["context_timestamps"][-1] + pd.Timedelta(hours=1),
                                       periods=prediction_length, freq="h")
        if not future_times.equals(expected_times):
            raise ValueError("future hours must continue the input reference grid, including its withheld gap")
        if not np.array_equal(record["class_target_hour_indices"], np.arange(prediction_length - 3, prediction_length)):
            raise ValueError("the class must use exactly the last three forecast hours")
        for values in record["past_covariates"].values():
            if np.shape(values) != history.shape or np.isinf(values).any():
                raise ValueError("past covariate length/value mismatch")
        for values in record["future_covariates"].values():
            if np.shape(values) != (prediction_length,) or not np.isfinite(values).all():
                raise ValueError("known future calendar length/value mismatch")
    return prediction_length


class Chronos2Backend:
    name = "chronos2"

    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata

    def forecast(self, records: list[dict]) -> np.ndarray:
        length = validate_records(records)
        inputs = [{
            "target": np.asarray(r["target"], dtype=np.float32),
            "past_covariates": {k: np.asarray(v, dtype=np.float32) for k, v in r["past_covariates"].items()},
            "future_covariates": {k: np.asarray(v, dtype=np.float32) for k, v in r["future_covariates"].items()},
        } for r in records]
        # Chronos counts target AND covariate rows in its internal batch size.
        rows_per_case = 1 + len(inputs[0]["past_covariates"])
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=length, quantile_levels=list(QUANTILE_LEVELS),
            batch_size=len(records) * rows_per_case, cross_learning=False,
            context_length=len(records[0]["target"]),
        )
        if len(quantiles) != len(records):
            raise ValueError("Chronos-2 returned the wrong number of cases")
        arrays = [_numpy(q) for q in quantiles]
        if any(q.shape != (1, length, len(QUANTILE_LEVELS)) for q in arrays):
            raise ValueError("unexpected Chronos-2 quantile axes; expected target/time/quantile")
        return check_forecasts(np.stack([q[0] for q in arrays]), len(records), length)


class TimestralBackend:
    name = "timestral"

    def __init__(self, forecaster, metadata: dict):
        self.forecaster = forecaster
        self.metadata = metadata

    def forecast(self, records: list[dict]) -> np.ndarray:
        length = validate_records(records)
        past_keys = sorted(set(records[0]["past_covariates"]) - set(CALENDAR_COLUMNS))
        context = np.stack([r["target"] for r in records]).astype(np.float32)[:, None, :]
        past = np.stack([[r["past_covariates"][k] for k in past_keys] for r in records]).astype(np.float32)
        # Timestral's known_covariates include BOTH context and forecast periods.
        known = np.stack([[
            np.concatenate([r["past_covariates"][k], r["future_covariates"][k]])
            for k in CALENDAR_COLUMNS
        ] for r in records]).astype(np.float32)
        forecast = self.forecaster.forecast(
            context, prediction_length=length, past_covariates=past, known_covariates=known,
        )
        levels = list(forecast.quantile_levels)
        try:
            indices = [next(i for i, value in enumerate(levels) if np.isclose(q, value)) for q in QUANTILE_LEVELS]
        except StopIteration as exc:
            raise ValueError("Timestral does not expose the required shared quantile levels") from exc
        quantiles = _numpy(forecast.quantiles)
        if quantiles.shape != (len(records), 1, length, len(levels)):
            raise ValueError("unexpected Timestral quantile axes; expected batch/target/time/quantile")
        return check_forecasts(np.take(quantiles[:, 0], indices, axis=-1), len(records), length)


def check_forecasts(values, n: int, prediction_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (n, prediction_length, len(QUANTILE_LEVELS)) or not np.isfinite(values).all():
        raise ValueError("expected finite batch/time/quantile predictions")
    return values


def forecast_features(backend, records: list[dict]) -> np.ndarray:
    """Keep the target hours separately; do not sum marginal quantiles."""
    length = validate_records(records)
    quantiles = check_forecasts(backend.forecast(records), len(records), length)
    target = np.stack([quantiles[i, r["class_target_hour_indices"], :] for i, r in enumerate(records)])
    return target.reshape(len(records), len(FEATURE_NAMES))


def load_backend(name: str, *, model_id: str, cache_dir: str | Path,
                 revision: str | None = None, device: str = "cpu", local_files_only: bool = False):
    """Download official weights only; native packages are imported lazily."""
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        model_id, revision=revision, cache_dir=str(cache_dir), local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "LICENSE*"],
    ))
    metadata = {
        "backend": name, "model_id": model_id, "model_revision": snapshot.name,
        "device": device, "quantile_levels": list(QUANTILE_LEVELS),
        "foundation_weights": "frozen; local inference only",
        "future_covariates": CALENDAR_COLUMNS,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "numpy", "huggingface_hub", "scikit-learn"]},
    }
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        metadata["versions"]["chronos-forecasting"] = importlib.metadata.version("chronos-forecasting")
        metadata["cross_learning"] = False
        pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map=device, local_files_only=True)
        return Chronos2Backend(pipeline, metadata)
    if name == "timestral":
        from timestral import TimestralForecaster
        metadata["versions"]["timestral"] = importlib.metadata.version("timestral")
        metadata["inference_options"] = "native default symmetric averaging and nonnegative clipping"
        forecaster = TimestralForecaster.from_pretrained(str(snapshot), device=device, local_files_only=True)
        return TimestralBackend(forecaster, metadata)
    raise ValueError(f"unknown backend: {name}")


def _transformed_features(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("expected 27 finite target-hour quantile features")
    # Keep negative model forecasts as information; do not silently call them dry.
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class ClassProbabilityHead:
    """Serializable multinomial logistic head; no pickle or foundation fine-tuning."""
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    training: dict

    @classmethod
    def fit(cls, x, y, cases: pd.DataFrame, *, regularization_c: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        if not len(cases) or not cases.split.eq("train").all():
            raise ValueError("the classification head may only fit on training cases")
        if cases.case_id.duplicated().any():
            raise ValueError("duplicate training case IDs")
        values = _transformed_features(x)
        y = np.asarray(y)
        if values.shape[0] != len(cases) or y.shape != (len(cases),) or set(np.unique(y)) != {0, 1, 2}:
            raise ValueError("training requires aligned cases and all three categorical classes")
        if not np.isfinite(regularization_c) or regularization_c <= 0:
            raise ValueError("regularization C must be positive")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        classifier = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=2000, random_state=0)
        classifier.fit(scaled, y.astype(int))
        if classifier.n_iter_.max() >= classifier.max_iter:
            raise RuntimeError("classification head did not converge")
        if not np.array_equal(classifier.classes_, [0, 1, 2]):
            raise ValueError("unexpected class probability order")
        training = {
            "split": "train", "n": len(y), "regularization_c": regularization_c,
            "first_issue_time": str(cases.issue_time.min()), "last_issue_time": str(cases.issue_time.max()),
            "last_target_end": str(cases.target_end_inclusive.max()),
            "class_counts": np.bincount(y.astype(int), minlength=3).tolist(),
            "class_weight": None, "calibration": "not separately calibrated; requires held-out assessment",
        }
        return cls(scaler.mean_, scaler.scale_, classifier.coef_, classifier.intercept_, training)

    def predict_proba(self, x):
        x = (_transformed_features(x) - self.mean) / self.scale
        logits = x @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def save(self, path: str | Path):
        state = {
            "format": "precipitation_quantile_classifier_v1", "classes": [0, 1, 2],
            "feature_names": FEATURE_NAMES, "transform": "sign(x) * log1p(abs(x)), then train-only StandardScaler",
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(), "intercept": self.intercept.tolist(),
            "training": self.training,
        }
        Path(path).write_text(json.dumps(state, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        state = json.loads(Path(path).read_text())
        if (state.get("format") != "precipitation_quantile_classifier_v1"
                or state.get("classes") != [0, 1, 2] or state.get("feature_names") != FEATURE_NAMES):
            raise ValueError("unsupported classifier format, class order or feature schema")
        arrays = [np.asarray(state[k], dtype=float) for k in ("mean", "scale", "coefficients", "intercept")]
        if ([a.shape for a in arrays] != [(27,), (27,), (3, 27), (3,)]
                or not all(np.isfinite(a).all() for a in arrays) or (arrays[1] <= 0).any()):
            raise ValueError("invalid classifier parameters")
        return cls(*arrays, state["training"])
