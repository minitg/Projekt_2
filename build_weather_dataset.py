from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLE_LABELS = {
    "TT_TU": "Lufttemperatur [C]",
    "RF_TU": "Relative Feuchte [%]",
    "R1": "Niederschlag [mm]",
    "SD_SO": "Sonnenschein",
    "F": "Windgeschwindigkeit [m/s]",
    "P0": "Luftdruck Stationshoehe [hPa]",
}


def find_product_files(data_root: Path) -> dict[str, Path]:
    product_files = {}
    for path in sorted(data_root.glob("stundenwerte_*_*/produkt_*_stunde_*.txt")):
        code = path.name.split("_")[1].upper()
        product_files[code] = path
    return product_files


def load_product(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        na_values=[-999, -999.0],
        skipinitialspace=True,
        low_memory=False,
    )
    df.columns = [col.strip() for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("eor")])
    df["datetime"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H")
    df = df.drop(columns=["MESS_DATUM"])
    return df.set_index("datetime").sort_index()


def select_columns(datasets: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    pieces = []
    for df in datasets.values():
        available = [col for col in columns if col in df.columns]
        if available:
            pieces.append(df[available])
    if not pieces:
        raise ValueError("None of the requested weather columns were found.")
    return pd.concat(pieces, axis=1).loc[:, lambda frame: ~frame.columns.duplicated()]


def build_weather_dataset(
    data_root: Path = Path("XAI_Wetter"),
    output_dir: Path = Path("data/processed"),
    start: str | None = "2015-01-01",
    end: str | None = "2024-12-31 23:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_files = find_product_files(data_root)
    datasets = {code: load_product(path) for code, path in product_files.items()}

    hourly = select_columns(datasets, list(VARIABLE_LABELS))
    if start:
        hourly = hourly.loc[pd.to_datetime(start) :]
    if end:
        hourly = hourly.loc[: pd.to_datetime(end)]

    daily = hourly.resample("D").agg(
        {
            "TT_TU": "mean",
            "RF_TU": "mean",
            "R1": "sum",
            "SD_SO": "sum",
            "F": "mean",
            "P0": "mean",
        }
    )
    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_cols] = daily[numeric_cols].interpolate(
        method="time", limit_direction="both"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_dir / "weather_clean_hourly.csv")
    daily.to_csv(output_dir / "weather_clean_daily.csv")
    return hourly, daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean weather CSVs from DWD files.")
    parser.add_argument("--data-root", type=Path, default=Path("XAI_Wetter"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31 23:00")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, daily_data = build_weather_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
    )
    print(f"Wrote {len(daily_data)} daily rows to {args.output_dir}")
