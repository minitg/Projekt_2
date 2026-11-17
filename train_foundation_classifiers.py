"""Train/evaluate supervised precipitation adapters on frozen local forecasters."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from classification_data import file_sha256, load_classification_bundle
from evaluate_precipitation_baselines import probability_metrics
from foundation_classifiers import (
    ClassProbabilityHead, FEATURE_NAMES, forecast_features, load_backend,
)


ROOT = Path(__file__).resolve().parent


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selected_cases(data, horizon: int, split: str, max_cases: int | None):
    cases = data.select(horizon, split)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max cases must be positive")
        # Uniformly cover the period, independent of class labels, including eval.
        if len(cases) > max_cases:
            positions = np.linspace(0, len(cases) - 1, max_cases).round().astype(int)
            cases = cases.iloc[positions].reset_index(drop=True)
    return cases


def extract_split_features(data, backend, horizon: int, split: str, *, cache_dir: Path,
                           dataset_fingerprint: str, batch_size: int = 8,
                           max_cases: int | None = None, refresh: bool = False):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    cases = selected_cases(data, horizon, split, max_cases)
    if cases.empty:
        raise ValueError(f"no eligible cases for horizon={horizon}, split={split}")
    identity = {
        "dataset_fingerprint": dataset_fingerprint, "backend": backend.metadata,
        "case_ids": cases.case_id.tolist(), "feature_names": FEATURE_NAMES,
        "adapter_code_sha256": file_sha256(ROOT / "foundation_classifiers.py"),
        "runner_code_sha256": file_sha256(Path(__file__)),
    }
    key = _digest(identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{backend.name}_h{horizon:02d}_{split}_{key[:16]}.npz"
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved["identity"].item() != key or saved["case_ids"].tolist() != cases.case_id.tolist():
                raise ValueError("forecast cache belongs to different cases or model settings")
            x = saved["features"].copy()
        if x.shape != (len(cases), len(FEATURE_NAMES)) or not np.isfinite(x).all():
            raise ValueError("invalid forecast feature cache; regenerate with --refresh-features")
        print(f"{backend.name} h={horizon} {split}: loaded {len(cases)} cached forecasts", flush=True)
        return x, cases, cache

    iterator = replace(data, cases=cases).foundation_records(horizon, split)
    chunks = []
    consumed_ids = []
    start = time.monotonic()
    while batch := list(islice(iterator, batch_size)):
        chunks.append(forecast_features(backend, batch))
        consumed_ids.extend(record["case_id"] for record in batch)
        if len(chunks) == 1 or len(chunks) % 20 == 0 or len(consumed_ids) == len(cases):
            print(f"{backend.name} h={horizon} {split}: {len(consumed_ids)}/{len(cases)} ({time.monotonic() - start:.1f}s)", flush=True)
    if consumed_ids != cases.case_id.tolist():
        raise ValueError("forecast case order differs from the classification dataset")
    x = np.concatenate(chunks)
    np.savez_compressed(cache, identity=np.array(key), case_ids=cases.case_id.to_numpy(dtype=str), features=x)
    return x, cases, cache


def aligned_labels(data, cases) -> np.ndarray:
    labels = data.labels.set_index("case_id").loc[cases.case_id, "target_class"]
    if labels.isna().any():
        raise ValueError("missing observed classification labels")
    return labels.to_numpy(dtype=np.int64)


def classification_metrics(y, p) -> dict:
    result = probability_metrics(y, p)
    predicted = p.argmax(axis=1)
    result["log_loss"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    result["confusion_matrix_true_rows_predicted_columns"] = matrix.tolist()
    for k in range(3):
        result[f"class{k}_precision"] = float(matrix[k, k] / matrix[:, k].sum()) if matrix[:, k].sum() else 0.0
    return result


def train_and_evaluate(data, backend, output_dir: Path, *, dataset_fingerprint: str,
                       horizons: list[int], batch_size: int = 8, max_cases: int | None = None,
                       regularization_c: float = 1.0, refresh: bool = False) -> list[dict]:
    if len(set(horizons)) != len(horizons) or not set(horizons).issubset(data.config.horizons_hours):
        raise ValueError("choose distinct horizons available in the dataset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in horizons:
        directory = output_dir / f"h{horizon:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        train_x, train_cases, train_cache = extract_split_features(
            data, backend, horizon, "train", cache_dir=output_dir / "feature_cache",
            dataset_fingerprint=dataset_fingerprint, batch_size=batch_size,
            max_cases=max_cases, refresh=refresh,
        )
        train_y = aligned_labels(data, train_cases)
        head = ClassProbabilityHead.fit(train_x, train_y, train_cases, regularization_c=regularization_c)
        head.save(directory / "classifier.json")
        # Verify that the persisted classifier is the one used for evaluation.
        head = ClassProbabilityHead.load(directory / "classifier.json")
        prior = (np.bincount(train_y, minlength=3) + 1) / (len(train_y) + 3)
        fitted_head_sha256 = file_sha256(directory / "classifier.json")
        predictions = []
        run_rows = []
        feature_caches = {"train": str(train_cache)}
        for split in ("validation", "evaluation"):
            x, cases, cache = extract_split_features(
                data, backend, horizon, split, cache_dir=output_dir / "feature_cache",
                dataset_fingerprint=dataset_fingerprint, batch_size=batch_size,
                max_cases=max_cases, refresh=refresh,
            )
            p = head.predict_proba(x)
            y = aligned_labels(data, cases)
            # Evaluation labels have not entered the head or its scaler.
            if file_sha256(directory / "classifier.json") != fitted_head_sha256:
                raise RuntimeError("the fitted head changed during evaluation")
            metrics = classification_metrics(y, p)
            reference = probability_metrics(y, np.tile(prior, (len(y), 1)))
            metrics.update({
                "model": backend.name, "model_id": backend.metadata["model_id"],
                "horizon_h": horizon, "split": split,
                "brier_skill_vs_training_frequency": 1 - metrics["brier_multiclass"] / reference["brier_multiclass"],
                "training_frequency_brier": reference["brier_multiclass"],
                "subset_run": max_cases is not None,
            })
            run_rows.append(metrics)
            frame = cases.copy()
            frame["target_class"] = y
            frame["model"] = backend.name
            for k in range(3):
                frame[f"p_class{k}"] = p[:, k]
            frame["predicted_class"] = p.argmax(axis=1)
            predictions.append(frame)
            feature_caches[split] = str(cache)
            print(f"{backend.name} h={horizon} {split}: Brier={metrics['brier_multiclass']:.5f}, Macro-F1={metrics['macro_f1']:.5f}", flush=True)
        pd.concat(predictions, ignore_index=True).to_parquet(directory / "predictions.parquet", index=False)
        (directory / "scores.json").write_text(json.dumps(run_rows, indent=2) + "\n")
        run = {
            "experiment": "frozen numerical forecaster + supervised multinomial logistic classification head",
            "zero_shot_classification": False, "backend": backend.metadata,
            "dataset_manifest_sha256": dataset_fingerprint,
            "target_class_threshold_mm": data.config.upper_class_threshold_mm,
            "horizon_h": horizon, "lookback_hours": data.config.lookback_hours,
            "observation_delay_hours": data.config.observation_delay_hours,
            "feature_names": FEATURE_NAMES, "head_training": head.training,
            "classifier_sha256": fitted_head_sha256, "feature_cache": feature_caches,
            "head_class_probability_order": [0, 1, 2],
            "max_cases_per_split": max_cases,
            "case_selection": "all eligible cases" if max_cases is None else "evenly spaced chronological subset, independent of labels; technical subset run, not a full benchmark",
            "validation_use": "reporting only; no tuning or recalibration",
            "evaluation_use": "retrospective 2024 reporting only; already explored, not an untouched test",
            "pretraining_overlap": "not independently excluded; results do not prove unseen station data during pretraining",
            "code_sha256": {name: file_sha256(ROOT / name) for name in
                            ["foundation_classifiers.py", "train_foundation_classifiers.py", "classification_data.py"]},
        }
        (directory / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        rows.extend(run_rows)
    (output_dir / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/precipitation_classification_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/foundation_classifiers")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/foundation_models")
    parser.add_argument("--backends", nargs="+", choices=["chronos2", "timestral"], default=["chronos2", "timestral"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3])
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--timestral-model", default="mistralai/Timestral-5M")
    parser.add_argument("--chronos-revision")
    parser.add_argument("--timestral-revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="forecast cases per batch, not covariate rows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-cases-per-split", type=int, help="technical subset run; omit for a full run")
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch size must be positive")
    if args.max_cases_per_split is not None and args.max_cases_per_split < 1:
        parser.error("max cases must be positive")
    if not np.isfinite(args.regularization_c) or args.regularization_c <= 0:
        parser.error("regularization C must be positive")
    data = load_classification_bundle(args.data_dir)
    if len(set(args.horizons)) != len(args.horizons) or not set(args.horizons).issubset(data.config.horizons_hours):
        parser.error("choose distinct horizons available in the dataset")
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    fingerprint = file_sha256(args.data_dir / "manifest.json")
    run_type = "full" if args.max_cases_per_split is None else f"subset_{args.max_cases_per_split}"
    for name in dict.fromkeys(args.backends):
        model_id = args.chronos_model if name == "chronos2" else args.timestral_model
        revision = args.chronos_revision if name == "chronos2" else args.timestral_revision
        print(f"Loading {model_id} on {args.device}", flush=True)
        backend = load_backend(name, model_id=model_id, cache_dir=args.cache_dir,
                               revision=revision, device=args.device, local_files_only=args.local_files_only)
        # Different checkpoints and subset/full runs never overwrite each other.
        model_key = _digest({"model_id": model_id, "revision": backend.metadata["model_revision"]})[:12]
        output = args.output_dir / f"{name}_{model_key}" / run_type
        train_and_evaluate(
            data, backend, output, dataset_fingerprint=fingerprint, horizons=args.horizons,
            batch_size=args.batch_size, max_cases=args.max_cases_per_split,
            regularization_c=args.regularization_c, refresh=args.refresh_features,
        )
        print(f"Saved classifier, probabilities and metrics to {output}", flush=True)
        del backend


if __name__ == "__main__":
    main()
