"""Verify the project's hourly CSV against original station files and time metadata.

The optional 10-minute archives are primary DWD data used to estimate, not to
certify, the minute-level reference of R1. No model targets are replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GROUPS = {"TU": (["TT_TU", "RF_TU"], "QN_9"), "RR": (["R1"], "QN_8"),
          "SD": (["SD_SO"], "QN_7"), "FF": (["F"], "QN_3"), "P0": (["P0"], "QN_8")}
HOURLY_DOC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf"
TEN_MIN_DOC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/BESCHREIBUNG_obsgermany-climate-10min-precipitation_de.pdf"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_metadata_rows(path, columns, start, end):
    rows = []
    with path.open(encoding="latin1") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if (row.get("Stations_ID") or "").strip() != "4104" or row.get("Parameter") not in columns:
                continue
            begin = pd.to_datetime(row["Von_Datum"], format="%Y%m%d")
            finish = pd.to_datetime(row["Bis_Datum"], format="%Y%m%d") + pd.Timedelta(days=1)
            if begin <= end and finish > start:
                if "UTC" not in row.get("Zusatz-Info", ""):
                    raise ValueError(f"non-UTC metadata period for {row['Parameter']}: {row}")
                rows.append({"parameter": row["Parameter"], "begin": str(begin), "end_exclusive": str(finish),
                             "time_description": row["Zusatz-Info"], "source": row["Datenquelle (Strukturversion=SV)"]})
    for col in columns:
        covered_until = start
        for row in sorted((r for r in rows if r["parameter"] == col), key=lambda r: r["begin"]):
            if pd.Timestamp(row["begin"]) > covered_until:
                break
            covered_until = max(covered_until, pd.Timestamp(row["end_exclusive"]))
        if covered_until <= end:
            raise ValueError(f"metadata does not establish uninterrupted UTC coverage for {col}")
    return rows


def compare_ten_minute_archives(archives, source, output):
    parts, provenance = [], []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.startswith("produkt") and n.endswith(".txt")]
            if len(members) != 1:
                raise ValueError("expected exactly one DWD ten-minute product")
            with archive.open(members[0]) as handle:
                frame = pd.read_csv(handle, sep=";", skipinitialspace=True, na_values=[-999])
            if not frame.STATIONS_ID.eq(4104).all():
                raise ValueError("wrong station in ten-minute archive")
            frame.index = pd.to_datetime(frame.MESS_DATUM, format="%Y%m%d%H%M")
            parts.append(frame.RWS_10)
        period = re.search(r"_(\d{8})_(\d{8})_04104\.txt$", members[0])
        if period is None:
            raise ValueError("unexpected ten-minute product naming")
        canonical = f"10minutenwerte_nieder_04104_{period[1]}_{period[2]}_hist.zip"
        cache = ROOT / "data/raw_audit" / canonical
        cache.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != cache.resolve():
            shutil.copyfile(path, cache)
        provenance.append({"file": canonical, "cached_file": str(cache.relative_to(ROOT)),
                           "sha256": sha(path), "product": members[0],
                           "url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/historical/" + canonical})
    series = pd.concat(parts).sort_index()
    if series.index.duplicated().any():
        raise ValueError("overlapping ten-minute archives need an explicit conflict policy")
    series = series.asfreq("10min")
    rolling = series.rolling(6, min_periods=6).sum().round(1)
    source = source.loc[max(source.index.min(), series.index.min()):min(source.index.max(), series.index.max())]
    candidates = {offset: pd.Series(rolling.reindex(source.index + pd.to_timedelta(offset, unit="min")).to_numpy(), index=source.index)
                  for offset in range(-60, 61, 10)}
    # Use identical support for every offset within each year.
    common = source.R1.notna()
    for candidate in candidates.values():
        common &= candidate.notna()
    rows = []
    for year in sorted(source.index.year.unique()):
        selected = common & (source.index.year == year)
        if not selected.any():
            continue
        for offset, candidate in candidates.items():
            errors = (source.R1 - candidate).abs()
            wet = selected & (source.R1.gt(0) | candidate.gt(0))
            rows.append({"year": int(year), "end_offset_minutes": offset, "n_common": int(selected.sum()),
                         "mae_mm": float(errors[selected].mean()),
                         "exact_fraction": float(errors[selected].lt(1e-6).mean()),
                         "wet_n": int(wet.sum()), "wet_mae_mm": float(errors[wet].mean())})
    result = pd.DataFrame(rows)
    result.to_csv(output / "rain_interval_comparison.csv", index=False)
    if result.empty:
        raise ValueError("no comparable ten-minute data")
    best = result.loc[result.groupby("year").mae_mm.idxmin()]
    if not best.end_offset_minutes.eq(-10).all():
        raise ValueError("observed reference offset is not consistently -10 minutes; inspect comparison")
    return provenance, best.to_dict("records")


def audit(raw_root: Path, source: Path, output: Path, ten_minute_archives):
    source_frame = pd.read_csv(source, parse_dates=["datetime"]).set_index("datetime").sort_index()
    start, end = source_frame.index.min(), source_frame.index.max()
    output.mkdir(parents=True, exist_ok=True)
    comparisons, metadata, raw_files = [], [], []
    quality = pd.DataFrame(index=source_frame.index)
    quality_rows = []
    for code, (columns, qn) in GROUPS.items():
        directories = sorted(raw_root.glob(f"stundenwerte_{code}_04104_*_hist"))
        if len(directories) != 1:
            raise ValueError(f"expected one historical directory for {code}")
        directory = directories[0]
        product = next(directory.glob("produkt*.txt"))
        selected_columns = {"STATIONS_ID", "MESS_DATUM", qn, *columns}
        raw = pd.read_csv(product, sep=";", encoding="latin1", skipinitialspace=True,
                          usecols=lambda name: name.strip() in selected_columns, na_values=[-999])
        raw.columns = raw.columns.str.strip()
        if not raw.STATIONS_ID.eq(4104).all():
            raise ValueError(f"wrong station in {product}")
        raw.index = pd.to_datetime(raw.MESS_DATUM, format="%Y%m%d%H")
        if raw.index.duplicated().any():
            raise ValueError(f"duplicate original timestamps in {product}")
        raw = raw.reindex(source_frame.index)
        for col in columns:
            known = source_frame[col].notna() & raw[col].notna()
            different = known & ~np.isclose(source_frame[col], raw[col], atol=1e-9, rtol=0)
            missing_difference = source_frame[col].isna() != raw[col].isna()
            comparisons.append({"parameter": col, "rows": len(raw), "known_compared": int(known.sum()),
                                "value_mismatches": int(different.sum()), "missing_mask_mismatches": int(missing_difference.sum())})
            if different.any() or missing_difference.any():
                raise ValueError(f"CSV does not match original {col}")
            quality[f"{col}_QN"] = raw[qn].astype("Int16")
            for level, count in raw[qn].value_counts(dropna=False).items():
                quality_rows.append({"parameter": col, "quality_level": int(level) if pd.notna(level) else "missing", "hours": int(count)})
        parameter = next(directory.glob("Metadaten_Parameter*.txt"))
        rows = utc_metadata_rows(parameter, columns, start, end)
        shutil.copyfile(parameter, output / parameter.name)
        metadata.append({"file": parameter.name, "sha256": sha(parameter), "verified_rows": rows})
        raw_files.append({"file": str(product.resolve()), "sha256": sha(product)})
    rr_dir = next(raw_root.glob("stundenwerte_RR_04104_*_hist"))
    for pattern in ["Metadaten_Stationsname*.txt", "Metadaten_Geraete_Niederschlagshoehe*.txt", "Metadaten_Geographie*.txt"]:
        for path in rr_dir.glob(pattern):
            shutil.copyfile(path, output / path.name)
    pd.DataFrame(comparisons).to_csv(output / "source_comparison.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output / "source_quality_summary.csv", index=False)
    quality.index = quality.index.tz_localize("UTC")
    quality.to_parquet(output / "source_quality_flags.parquet")
    archive_provenance, interval_evidence = compare_ten_minute_archives(ten_minute_archives, source_frame, output)
    report = {
        "station_id": "04104", "station_name": "Regensburg", "timezone": "UTC", "timezone_verified": True,
        "verified_reference_start": str(start), "verified_reference_end": str(end),
        "source_csv_sha256": sha(source), "source_comparison": comparisons,
        "metadata": metadata, "original_products": raw_files,
        "ten_minute_archives": archive_provenance, "interval_evidence_by_year": interval_evidence,
        "r1_reference_to_interval_end_minutes": -10,
        "interval_evidence_status": "empirical comparison with primary DWD 10-minute data; not a formal minute-level guarantee",
        "recommended_primary_start": "2018-01-01",
        "primary_start_reason": "rain[e]H3 installed 2017-11-14; use complete years with one precipitation instrument generation",
        "previous_instrument": "PLUVIO through 2017-11-13; DWD describes approximately 5 additional minutes of response lag",
        "hourly_recent_publication": "daily, rolling 500 days through yesterday, according to DWD documentation",
        "hourly_historical_publication": "annually updated, quality-checked historical version",
        "one_hour_availability_verified": False,
        "point_in_time_versions_available": False,
        "study_mode": "retrospective finalized-data benchmark under explicitly hypothetical observation delays",
        "official_sources": [HOURLY_DOC, TEN_MIN_DOC],
        "audit_code_sha256": sha(Path(__file__)),
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(pd.DataFrame(comparisons).to_string(index=False))
    print("\nBest interval alignment by year:\n" + pd.DataFrame(interval_evidence).round(5).to_string(index=False))
    print(f"\nSaved evidence to {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/weather_clean_hourly.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/provenance/dwd_04104")
    parser.add_argument("--ten-minute-archive", type=Path, action="append", required=True)
    args = parser.parse_args()
    audit(args.raw_root, args.source, args.output_dir, args.ten_minute_archive)


if __name__ == "__main__":
    main()
