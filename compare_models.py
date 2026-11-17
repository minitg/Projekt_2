"""Merge TiViT, Chronos-2, Timestral (and later TTM) scores into one table on identical cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ["model", "horizon_h", "split", "n", "accuracy", "balanced_accuracy", "macro_f1",
           "brier_multiclass", "brier_wet", "encode_seconds", "inference_ms_per_case",
           "head_fit_seconds", "device", "torch_threads", "batch_size"]


def collect(outputs: Path, run_type: str, horizon: int):
    """Return (scores, {(model, split): case_ids}) from every known runner layout."""
    scores, cases = [], {}
    for directory in sorted((outputs / "tivit_clip_b32").glob(run_type)):
        frame = pd.read_csv(directory / "scores.csv")
        scores.append(frame[frame.horizon_h == horizon])
        for split in ("validation", "evaluation"):
            ids = pd.read_parquet(directory / f"{split}_predictions.parquet", columns=["case_id"]).case_id
            cases[("TiViT-CLIP-B32", split)] = set(ids)
    for directory in sorted((outputs / "foundation_classifiers").glob(f"*/{run_type}/h{horizon:02d}")):
        scores.append(pd.DataFrame(json.loads((directory / "scores.json").read_text())))
        predictions = pd.read_parquet(directory / "predictions.parquet", columns=["case_id", "split", "model"])
        for (model, split), group in predictions.groupby(["model", "split"]):
            cases[(model, split)] = set(group.case_id)
    if not scores:
        raise FileNotFoundError(f"no model outputs for run type {run_type!r} and horizon {horizon} in {outputs}")
    return pd.concat(scores, ignore_index=True), cases


def compare(outputs: Path, run_type: str, horizon: int) -> pd.DataFrame:
    scores, cases = collect(outputs, run_type, horizon)
    for split in scores.split.unique():
        sets = {model: ids for (model, s), ids in cases.items() if s == split}
        if len({frozenset(ids) for ids in sets.values()}) > 1:
            raise ValueError(f"models were not evaluated on identical {split} cases: "
                             f"{ {m: len(i) for m, i in sets.items()} }")
    if scores.groupby("split").device.nunique().gt(1).any():
        print("WARNING: models ran on different devices; inference times are not directly comparable")
    return scores.reindex(columns=COLUMNS).sort_values(["split", "brier_multiclass"], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-type", default="full", help="'full' or 'subset_<N>'")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    table = compare(args.outputs, args.run_type, args.horizon)
    target = args.outputs / f"comparison_h{args.horizon:02d}_{args.run_type}.csv"
    table.to_csv(target, index=False)
    print(table.to_string(index=False))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
