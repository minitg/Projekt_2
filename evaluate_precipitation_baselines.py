"""Frozen count-based baselines on exactly the precipitation_v2 eligible cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from precipitation_data import load_precipitation_bundle

ROOT = Path(__file__).resolve().parent


def probability_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (len(y), 3) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError("expected three finite, nonnegative probabilities summing to one")
    if not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("expected observed target classes 0, 1 or 2")
    predicted = p.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    return {
        "n": len(y), "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
        "brier_multiclass": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
        "brier_wet": float(np.square(p[:, 1:].sum(axis=1) - (y > 0)).mean()),
        **{f"class{k}_support": int(support[k]) for k in range(3)},
        **{f"class{k}_recall": float(recall[k]) for k in range(3)},
    }


def latest_observed_class(observations, cases, threshold):
    """Class of the last three observed hours available before issuance.

    Unknown input blocks have state 3; they are never silently called dry.
    """
    values = np.column_stack([
        observations.R1.reindex(pd.DatetimeIndex(cases.input_end) - pd.to_timedelta(j, unit="h")).to_numpy()
        for j in [2, 1, 0]
    ])
    valid = np.isfinite(values).all(axis=1)
    amount = np.nansum(values, axis=1).round(1)
    return np.where(valid, np.where(amount == 0, 0, np.where(amount < threshold, 1, 2)), 3)


def fit_count_baselines(training, states):
    y = training.target_class.to_numpy(dtype=int)
    prior = (np.bincount(y, minlength=3) + 1) / (len(y) + 3)
    transitions = np.zeros((4, 3), dtype=int)
    np.add.at(transitions, (states, y), 1)
    transition_probs = (transitions + 24 * prior) / (transitions.sum(axis=1, keepdims=True) + 24)
    seasonal = np.empty((12, 24, 3))
    times = training.target_end_inclusive
    for month in range(1, 13):
        for hour in range(24):
            selected = y[(times.dt.month == month) & (times.dt.hour == hour)]
            seasonal[month - 1, hour] = (np.bincount(selected, minlength=3) + 24 * prior) / (len(selected) + 24)
    return {"prior": prior, "transitions": transition_probs, "seasonal": seasonal}


def predict_count_baselines(fitted, cases, states):
    prior = np.repeat(fitted["prior"][None, :], len(cases), axis=0)
    persistence = prior.copy()
    known = states < 3
    persistence[known] = np.eye(3)[states[known]]
    times = cases.target_end_inclusive
    return {
        "always_dry": np.tile([1., 0., 0.], (len(cases), 1)),
        "training_frequency": prior,
        "month_and_hour": fitted["seasonal"][times.dt.month.to_numpy() - 1, times.dt.hour.to_numpy()],
        "last_observed_class": persistence,
        "transition_frequency": fitted["transitions"][states],
    }


def evaluate_baselines(data_dir: Path, output: Path):
    bundle = load_precipitation_bundle(data_dir)
    observations = pd.read_parquet(data_dir / "hourly_observations.parquet")
    source_manifest = json.loads((data_dir / "manifest.json").read_text())
    rows, predictions, fitted_json = [], [], {}
    for horizon in bundle.config.horizons_hours:
        training = bundle.select(horizon, "train")
        train_states = latest_observed_class(observations, training, bundle.config.upper_class_threshold_mm)
        fitted = fit_count_baselines(training, train_states)
        fitted_json[str(horizon)] = {k: v.tolist() for k, v in fitted.items()}
        for split in ["validation", "evaluation"]:
            cases = bundle.select(horizon, split)
            states = latest_observed_class(observations, cases, bundle.config.upper_class_threshold_mm)
            y = cases.target_class.to_numpy(dtype=int)
            forecasts = predict_count_baselines(fitted, cases, states)
            reference = probability_metrics(y, forecasts["training_frequency"])["brier_multiclass"]
            for name, p in forecasts.items():
                scores = probability_metrics(y, p)
                rows.append({"horizon_h": horizon, "split": split, "baseline": name, **scores,
                             "brier_skill_vs_training_frequency": 1 - scores["brier_multiclass"] / reference,
                             "unknown_last_block": int((states == 3).sum())})
                result = cases[["issue_time", "issue_reference_time", "input_end", "target_end_inclusive", "target_reference_end", "horizon_h", "split", "target_class"]].copy()
                result["baseline"] = name
                result[["p_dry", "p_wet_below_threshold", "p_wet_above_threshold"]] = p
                predictions.append(result)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "fitted_baselines.json").write_text(json.dumps({
        "source_csv_sha256": source_manifest["source_sha256"],
        "dataset_config": source_manifest["config"],
        "fit_split": "train only; no validation or evaluation tuning",
        "prior_smoothing": "one pseudocount per class",
        "conditional_smoothing": "24 prior-equivalent observations, fixed in advance",
        "unknown_input_state": "state 3 in transition table; persistence falls back to training prior",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fitted": fitted_json,
    }, indent=2) + "\n")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/precipitation_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/precipitation_baselines_v2")
    args = parser.parse_args()
    scores = evaluate_baselines(args.data_dir, args.output_dir)
    columns = ["horizon_h", "split", "baseline", "n", "accuracy", "brier_multiclass", "brier_skill_vs_training_frequency"]
    print(scores[columns].round(5).to_string(index=False))
    print(f"\nSaved scores and per-case probabilities to {args.output_dir}")


if __name__ == "__main__":
    main()
