from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class WeatherDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_stats: dict[str, float]
    feature_stats: dict[str, dict[str, float]]
    train_samples: int
    val_samples: int
    test_samples: int


class WeatherSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        target_dates: np.ndarray,
        sequence_length: int,
        horizon: int,
        end_indices: np.ndarray,
    ) -> None:
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.target_dates = target_dates
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.indices = end_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        end = int(self.indices[item])
        start = end - self.sequence_length + 1
        target_idx = end + self.horizon
        x = self.features[start : end + 1]
        y = self.target[target_idx]
        date = str(self.target_dates[target_idx])
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32), date


DEFAULT_ROLLING_COLUMNS = ["TT_TU", "RF_TU", "R1", "SD_SO", "F", "P0"]
DEFAULT_ROLLING_WINDOWS = [3, 7, 14]
DEFAULT_LOG_COLUMNS = ["R1", "SD_SO", "F"]


def load_daily_weather(
    data_path: str | Path,
    target: str,
    add_engineered_features: bool = True,
    rolling_windows: list[int] | tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS),
    rolling_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_ROLLING_COLUMNS),
    log_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_LOG_COLUMNS),
) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime").asfreq("D")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric:
        raise ValueError(f"Target {target!r} not found in numeric columns: {numeric}")

    # Time-based interpolation keeps the daily grid complete for sequence windows.
    df[numeric] = df[numeric].interpolate(method="time", limit_direction="both")
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    if add_engineered_features:
        df = add_weather_features(
            df,
            rolling_windows=rolling_windows,
            rolling_columns=rolling_columns,
            log_columns=log_columns,
        )
    return df


def add_weather_features(
    df: pd.DataFrame,
    rolling_windows: list[int] | tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS),
    rolling_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_ROLLING_COLUMNS),
    log_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_LOG_COLUMNS),
) -> pd.DataFrame:
    engineered = df.copy()

    for col in rolling_columns:
        if col not in engineered.columns:
            continue
        for window in rolling_windows:
            if window <= 1:
                continue
            engineered[f"{col}_roll{window}_mean"] = (
                engineered[col].rolling(window=window, min_periods=1).mean()
            )

    for col in log_columns:
        if col not in engineered.columns:
            continue
        if engineered[col].min(skipna=True) < 0:
            continue
        engineered[f"{col}_log1p"] = np.log1p(engineered[col])

    return engineered


def split_end_indices(
    dates: pd.DatetimeIndex,
    sequence_length: int,
    horizon: int,
    train_until: str,
    val_until: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_end = len(dates) - horizon - 1
    candidate_end_indices = np.arange(sequence_length - 1, max_end + 1)
    target_dates = dates[candidate_end_indices + horizon]

    train_mask = target_dates <= pd.Timestamp(train_until)
    val_mask = (target_dates > pd.Timestamp(train_until)) & (
        target_dates <= pd.Timestamp(val_until)
    )
    test_mask = target_dates > pd.Timestamp(val_until)
    return (
        candidate_end_indices[train_mask],
        candidate_end_indices[val_mask],
        candidate_end_indices[test_mask],
    )


def standardize_from_train(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    train_end_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, dict[str, float]]]:
    last_train_target_idx = int(train_end_indices.max() + 1)
    train_frame = df.iloc[: last_train_target_idx + 1]

    feature_mean = train_frame[feature_columns].mean()
    feature_std = train_frame[feature_columns].std().replace(0, 1)
    target_mean = float(train_frame[target].mean())
    target_std = float(train_frame[target].std() or 1.0)

    features = ((df[feature_columns] - feature_mean) / feature_std).to_numpy()
    target_values = ((df[target] - target_mean) / target_std).to_numpy()
    feature_stats = {
        col: {"mean": float(feature_mean[col]), "std": float(feature_std[col])}
        for col in feature_columns
    }
    target_stats = {"mean": target_mean, "std": target_std}
    return features, target_values, target_stats, feature_stats


def make_weather_dataloaders(
    data_path: str | Path = "data/processed/weather_clean_daily.csv",
    target: str = "TT_TU",
    sequence_length: int = 30,
    horizon: int = 1,
    batch_size: int = 64,
    train_until: str = "2021-12-31",
    val_until: str = "2023-12-31",
    add_engineered_features: bool = False,
    rolling_windows: list[int] | tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS),
    rolling_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_ROLLING_COLUMNS),
    log_columns: list[str] | tuple[str, ...] = tuple(DEFAULT_LOG_COLUMNS),
) -> WeatherDataBundle:
    df = load_daily_weather(
        data_path,
        target,
        add_engineered_features=add_engineered_features,
        rolling_windows=rolling_windows,
        rolling_columns=rolling_columns,
        log_columns=log_columns,
    )
    feature_columns = df.columns.tolist()
    train_idx, val_idx, test_idx = split_end_indices(
        df.index, sequence_length, horizon, train_until, val_until
    )
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Empty train/validation/test split. Adjust split dates or horizon.")

    features, target_values, target_stats, feature_stats = standardize_from_train(
        df, feature_columns, target, train_idx
    )
    dates = df.index.strftime("%Y-%m-%d").to_numpy()

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = WeatherSequenceDataset(
            features=features,
            target=target_values,
            target_dates=dates,
            sequence_length=sequence_length,
            horizon=horizon,
            end_indices=indices,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return WeatherDataBundle(
        train_loader=make_loader(train_idx, shuffle=True),
        val_loader=make_loader(val_idx, shuffle=False),
        test_loader=make_loader(test_idx, shuffle=False),
        dataframe=df,
        feature_columns=feature_columns,
        target_stats=target_stats,
        feature_stats=feature_stats,
        train_samples=int(len(train_idx)),
        val_samples=int(len(val_idx)),
        test_samples=int(len(test_idx)),
    )
