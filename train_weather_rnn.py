from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import PaperLSTM
from weather_dataloader import (
    DEFAULT_LOG_COLUMNS,
    DEFAULT_ROLLING_COLUMNS,
    DEFAULT_ROLLING_WINDOWS,
    WeatherDataBundle,
    make_weather_dataloaders,
)


@dataclass
class Config:
    data_path: str = "data/processed/weather_clean_daily.csv"
    output_dir: str = "outputs/rnn_weather_paper_lstm"
    model: str = "paper_lstm"
    target: str = "TT_TU"
    sequence_length: int = 48
    horizon: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    train_until: str = "2021-12-31"
    val_until: str = "2023-12-31"
    seed: int = 42
    device: str = "auto"
    add_engineered_features: bool = False
    rolling_windows: tuple[int, ...] = tuple(DEFAULT_ROLLING_WINDOWS)
    rolling_columns: tuple[str, ...] = tuple(DEFAULT_ROLLING_COLUMNS)
    log_columns: tuple[str, ...] = tuple(DEFAULT_LOG_COLUMNS)


PAPER_LSTM_DEFAULTS = {
    "model": "paper_lstm",
    "sequence_length": 48,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.15,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "patience": 5,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_model(cfg: Config, n_features: int) -> nn.Module:
    if cfg.model == "paper_lstm":
        return PaperLSTM(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            dropout=cfg.dropout,
        )
    raise ValueError("Unknown model. Use 'paper_lstm'.")


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_stats: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    preds: list[float] = []
    actuals: list[float] = []
    dates: list[str] = []

    with torch.no_grad():
        for x, y, batch_dates in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(float(criterion(pred, y).item()))
            preds.extend(pred.cpu().numpy().reshape(-1).tolist())
            actuals.extend(y.cpu().numpy().reshape(-1).tolist())
            dates.extend(list(batch_dates))

    pred_arr = np.array(preds) * target_stats["std"] + target_stats["mean"]
    actual_arr = np.array(actuals) * target_stats["std"] + target_stats["mean"]
    frame = pd.DataFrame(
        {"datetime": pd.to_datetime(dates), "actual": actual_arr, "prediction": pred_arr}
    ).sort_values("datetime")
    return float(np.mean(losses)), frame


def train(cfg: Config) -> dict[str, object]:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = make_weather_dataloaders(
        data_path=cfg.data_path,
        target=cfg.target,
        sequence_length=cfg.sequence_length,
        horizon=cfg.horizon,
        batch_size=cfg.batch_size,
        train_until=cfg.train_until,
        val_until=cfg.val_until,
        add_engineered_features=cfg.add_engineered_features,
        rolling_windows=cfg.rolling_windows,
        rolling_columns=cfg.rolling_columns,
        log_columns=cfg.log_columns,
    )
    device = pick_device(cfg.device)
    model = make_model(cfg, n_features=len(data.feature_columns)).to(device)

    history, best_epoch = fit_model(model, data, cfg, device, output_dir)
    metrics = save_evaluation(model, data, cfg, device, output_dir, history, best_epoch)
    print(json.dumps(metrics, indent=2))
    return metrics


def fit_model(
    model: nn.Module,
    data: WeatherDataBundle,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, float]], int]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best_lstm.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, y, _ in data.train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, _ = evaluate(model, data.val_loader, criterion, device, data.target_stats)
        scheduler.step(val_loss)
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "feature_columns": data.feature_columns,
                    "feature_stats": data.feature_stats,
                    "target_stats": data.target_stats,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return history, best_epoch


def save_evaluation(
    model: nn.Module,
    data: WeatherDataBundle,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    history: list[dict[str, float]],
    best_epoch: int,
) -> dict[str, object]:
    criterion = nn.MSELoss()
    _, val_predictions = evaluate(model, data.val_loader, criterion, device, data.target_stats)
    _, test_predictions = evaluate(model, data.test_loader, criterion, device, data.target_stats)

    persistence = data.dataframe[cfg.target].shift(cfg.horizon).reindex(
        test_predictions["datetime"]
    )
    test_predictions["persistence"] = persistence.to_numpy()
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    metrics = {
        "best_epoch": best_epoch,
        "model": cfg.model,
        "target": cfg.target,
        "sequence_length": cfg.sequence_length,
        "horizon": cfg.horizon,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "patience": cfg.patience,
        "train_until": cfg.train_until,
        "val_until": cfg.val_until,
        "seed": cfg.seed,
        "add_engineered_features": cfg.add_engineered_features,
        "rolling_windows": list(cfg.rolling_windows),
        "rolling_columns": list(cfg.rolling_columns),
        "log_columns": list(cfg.log_columns),
        "feature_columns": data.feature_columns,
        "device": str(device),
        "train_samples": data.train_samples,
        "validation_samples": data.val_samples,
        "test_samples": data.test_samples,
        "validation_rmse": rmse(
            val_predictions["prediction"].to_numpy(), val_predictions["actual"].to_numpy()
        ),
        "validation_mae": mae(
            val_predictions["prediction"].to_numpy(), val_predictions["actual"].to_numpy()
        ),
        "test_rmse": rmse(
            test_predictions["prediction"].to_numpy(), test_predictions["actual"].to_numpy()
        ),
        "test_mae": mae(
            test_predictions["prediction"].to_numpy(), test_predictions["actual"].to_numpy()
        ),
        "persistence_test_rmse": rmse(
            test_predictions["persistence"].to_numpy(), test_predictions["actual"].to_numpy()
        ),
        "persistence_test_mae": mae(
            test_predictions["persistence"].to_numpy(), test_predictions["actual"].to_numpy()
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train an LSTM on daily weather data.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", choices=["paper_lstm"], default=Config.model)
    parser.add_argument("--target", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--train-until", default=None)
    parser.add_argument("--val-until", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--no-engineered-features",
        action="store_true",
        help="Disable rolling averages and log1p weather features.",
    )

    args = vars(parser.parse_args())
    cfg = Config(model=args.pop("model"))
    if cfg.model == "paper_lstm":
        for key, value in PAPER_LSTM_DEFAULTS.items():
            setattr(cfg, key, value)
        cfg.output_dir = "outputs/rnn_weather_paper_lstm"

    for key, value in args.items():
        if key == "no_engineered_features":
            if value:
                cfg.add_engineered_features = False
            continue
        if value is not None:
            setattr(cfg, key, value)
    return cfg


if __name__ == "__main__":
    train(parse_args())
