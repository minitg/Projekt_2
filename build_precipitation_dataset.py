"""Build precipitation_v2 with verified UTC and an explicit reference-time audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from precipitation_data import (
    PrecipitationConfig, WEATHER_COLUMNS, build_samples, prepare_features, read_observations,
    utc_timestamp,
)

ROOT = Path(__file__).resolve().parent


def build_dataset(source: Path, output: Path, cfg: PrecipitationConfig, metadata_audit: Path | None = None):
    cfg.validate()
    audit = None
    if metadata_audit is not None:
        audit = json.loads(metadata_audit.read_text())
        if (not audit.get("timezone_verified") or audit.get("timezone") != "UTC"
                or audit.get("source_csv_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
            raise ValueError("metadata audit does not verify UTC and this exact source CSV")
        if cfg.reference_offset_minutes != audit["r1_reference_to_interval_end_minutes"]:
            raise ValueError("configured interval offset differs from the primary-data audit")
    observations = read_observations(source)
    if cfg.source_start is not None:
        observations = observations.loc[utc_timestamp(cfg.source_start):].copy()
    if observations.empty:
        raise ValueError("no observations in the configured primary period")
    if metadata_audit is not None:
        quality = pd.read_parquet(metadata_audit.parent / "source_quality_flags.parquet")
        observations = observations.join(quality.reindex(observations.index))
    features, preprocessing = prepare_features(observations, cfg)
    samples = build_samples(observations, cfg)
    summary = []
    for (horizon, split), group in samples.groupby(["horizon_h", "split"], sort=True):
        good = group[group.eligible]
        counts = good.target_class.value_counts()
        summary.append({"horizon_h": int(horizon), "split": split, "candidates": len(group),
                        "eligible": len(good), "excluded": int((~group.eligible).sum()),
                        "incomplete_target": int((~group.target_complete).sum()),
                        "target_quality_excluded": int((group.target_complete & ~group.target_quality_acceptable).sum()),
                        "insufficient_history": int((~group.history_complete).sum()),
                        "split_boundary": int((~group.within_split).sum()),
                        "dry": int(counts.get(0, 0)), "wet_below_threshold": int(counts.get(1, 0)),
                        "wet_at_or_above_threshold": int(counts.get(2, 0))})
    summary = pd.DataFrame(summary)
    if any(summary.eligible.eq(0)):
        raise ValueError("at least one configured split/horizon has no eligible samples")
    mask_summary = pd.DataFrame([{
        "feature": col, "source_missing": int(observations[f"{col}_source_missing"].sum()),
        "invalid": int(observations[f"{col}_invalid"].sum()),
        "observed": int(observations[f"{col}_observed"].sum()),
    } for col in WEATHER_COLUMNS])
    output.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(output / "hourly_observations.parquet", engine="pyarrow")
    features.to_parquet(output / "hourly_features.parquet", engine="pyarrow")
    samples.to_parquet(output / "samples.parquet", engine="pyarrow", index=False)
    summary.to_csv(output / "split_summary.csv", index=False)
    mask_summary.to_csv(output / "observation_quality.csv", index=False)
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n")
    manifest = {
        "dataset_version": "precipitation_v2", "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "station_id": "04104", "station_id_provenance": "original product and station metadata audit" if audit else "unverified synthetic/API input",
        "source_kind": "existing hourly CSV with retained missingness; not a new raw-DWD import",
        "source_start": str(observations.index.min()), "source_end": str(observations.index.max()),
        "time_reference": "UTC, verified against station metadata" if audit else "UTC declared by caller; metadata audit not supplied",
        "metadata_audit_sha256": hashlib.sha256(metadata_audit.read_bytes()).hexdigest() if audit else None,
        "timezone_verified": bool(audit),
        "interval_end_reference_offset_minutes": cfg.reference_offset_minutes,
        "interval_end_evidence": audit["interval_evidence_status"] if audit else "caller-declared for synthetic/API input",
        "availability_verified": False,
        "operational_ready": False,
        "evaluation_status": "retrospective; 2024 was already explored in EDA",
        "target_contract": "three R1 reference hours ending at target_reference_end; estimated interval (issue_time+h-3,issue_time+h]",
        "issue_schedule": "DWD 3-hour reference grid plus configured minute offset; reference and planned issue time stored separately",
        "availability_assumption": "hypothetical delay from estimated interval end; daily CDC hourly archives are not a verified live feed",
        "config": asdict(cfg), "hourly_rows": len(observations), "feature_count": len(features.columns),
        "sample_rows": len(samples), "eligible_samples": int(samples.eligible.sum()),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ["precipitation_data.py", "build_precipitation_dataset.py"]},
        "versions": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "pyarrow"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "\n".join(f"| {r.horizon_h} | {r.split} | {r.eligible} | {r.excluded} | {r.dry} | {r.wet_below_threshold} | {r.wet_at_or_above_threshold} |"
                     for r in summary.itertuples())
    (output / "DATASET.md").write_text(f"""# Niederschlags-Datensatz v2

Retrospektiver Modellvergleich mit UTC-Zeitstempeln. Der Minutenbezug ist empirisch geprüft; reale Veröffentlichungslatenzen sind nicht nachgewiesen.

- Quelle: `{source.name}`, SHA-256 `{manifest['source_sha256']}`.
- Zeitraum: {manifest['source_start']} bis {manifest['source_end']}.
- Primärzeitraum ab {cfg.source_start}: vollständige Jahre nach dem Wechsel auf rain[e]H3.
- Historie: {cfg.lookback_hours} Stundenreferenzen; hypothetische Verzögerung {cfg.observation_delay_hours} Stunde(n) ab geschätztem Intervallende.
- Ausgabeplan: Dreistundenraster der Referenzstempel, verschoben um {cfg.reference_offset_minutes} Minuten. Der Standardplan liegt damit bei 23:50, 02:50, 05:50 usw. UTC.
- `input_end` und `target_reference_end` sind originale DWD-Referenzen; `issue_time` ist der geplante Ausgabezeitpunkt, `target_end_inclusive` das geschätzte Intervallende.
- Merkmale pro Stunde: {len(features.columns)}; gleiche Eingaben für LSTM und XGBoost.
- Ziel: drei vollständig beobachtete R1-Stunden; keine Zielimputation.
- Zugelassene DWD-Prüfverfahren für sämtliche Zielstunden: {cfg.accepted_target_quality_levels}. QN=3 steht für automatische Prüfung/Korrektur, QN=10 für abgeschlossene Prüfung/Korrektur. Diese Codes sind Verfahren, keine kontinuierliche Qualitätsskala. Beobachtete Zielwerte mit anderen Verfahren bleiben in der Prüftabelle, werden aber nicht zum Modelltraining verwendet.
- Klassen: 0 mm / mehr als 0 bis unter {cfg.upper_class_threshold_mm:g} mm / mindestens {cfg.upper_class_threshold_mm:g} mm.
- Training vor {cfg.train_end_exclusive}; Validierung danach bis vor {cfg.validation_end_exclusive}; spätere Daten zur rückblickenden Evaluierung.
- `{len(samples):,}` Kandidaten, `{int(samples.eligible.sum()):,}` nutzbare Vorhersagefälle über alle Vorläufe. Dieselben Zeiten kommen bei mehreren Vorläufen vor und sind keine unabhängigen Beobachtungen.
- Fehlende Niederschlags-/Sonnenstunden werden als Eingaben nicht fortgeschrieben. Trainingsmediane sind numerische Ersatzwerte mit separater Fehlwertkennzeichnung.
- Temperatur, Feuchte, Wind und Druck werden höchstens {cfg.forward_fill_limit_hours} Stunden aus der Vergangenheit fortgeschrieben; verbleibende Eingabelücken verwenden Trainingsmediane.
- Mittelwerte, Streuungen und Mediane stammen ausschließlich aus beobachteten Trainingswerten.

| Vorlauf h | Abschnitt | Nutzbar | Ausgeschlossen | Trocken | Nass unter Schwelle | Nass ab Schwelle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{rows}

`samples.parquet` enthält auch ausgeschlossene Fälle und ihre Gründe. `target_observed_hours` muss für nutzbare Fälle drei betragen. `hourly_observations.parquet` erhält Quellfehlwerte und Plausibilitätsmasken. `hourly_features.parquet` enthält numerische Modellinputs. `preprocessing.json` dokumentiert alle Transformationsparameter.

Die Vollständigkeit eines Historienfensters bezieht sich auf die Zeitachse. Einzelne fehlende Eingabemessungen bleiben über die Masken erkennbar. Ausgabe und geschätztes Zielintervall müssen im gleichen Abschnitt liegen. Historie aus einem früheren Abschnitt ist zulässig. Die Original-Qualitätsniveaus stehen bei einem verifizierten Build in `hourly_observations.parquet`; sie sind keine Modellmerkmale.

Die Klassen sind Projektdefinitionen, keine amtlichen Warnschwellen. Der DWD beschreibt das Stundenprodukt `recent` als täglich aktualisiert bis gestern. Die angenommene Stunde Verzögerung ist deshalb ein hypothetisches Experiment mit finalen Archivdaten; sie belegt keinen historischen Echtzeitbetrieb. Der frühere Parquet-Export mit globaler Imputation wird nicht verwendet.
""")
    return summary, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/weather_clean_hourly.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/precipitation_v2")
    parser.add_argument("--metadata-audit", type=Path, default=ROOT / "data/provenance/dwd_04104/audit.json")
    parser.add_argument("--source-start", default="2018-01-01")
    parser.add_argument("--accepted-target-quality", type=int, nargs="+", default=[3, 10])
    parser.add_argument("--lookback-hours", type=int, default=48)
    parser.add_argument("--observation-delay-hours", type=int, default=1)
    parser.add_argument("--train-end-exclusive", default="2022-01-01")
    parser.add_argument("--validation-end-exclusive", default="2024-01-01")
    args = parser.parse_args()
    cfg = PrecipitationConfig(lookback_hours=args.lookback_hours,
                             observation_delay_hours=args.observation_delay_hours,
                             train_end_exclusive=args.train_end_exclusive,
                             validation_end_exclusive=args.validation_end_exclusive,
                             source_start=args.source_start,
                             accepted_target_quality_levels=tuple(args.accepted_target_quality))
    summary, manifest = build_dataset(args.source, args.output_dir, cfg, metadata_audit=args.metadata_audit)
    print(summary.to_string(index=False))
    print(f"\nSaved {manifest['eligible_samples']:,} eligible examples to {args.output_dir}")


if __name__ == "__main__":
    main()
