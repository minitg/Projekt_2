# XAI Timeseries: Weather LSTM Forecasting

This repository contains a compact PyTorch pipeline for forecasting the next
day's mean temperature from DWD weather time series.

The final model is a WeatherLSTM architecture trained on daily weather data
from DWD station `04104` Regensburg.

## Structure

```text
data/
  processed/
    weather_clean_daily.csv
    weather_clean_hourly.csv
models/
  weather_lstm.py
notebooks/
  lstm_training.ipynb
build_weather_dataset.py
weather_dataloader.py
train_weather_rnn.py
requirements.txt
```

## Data

The model uses `data/processed/weather_clean_daily.csv`.

Important columns:

```text
datetime  Date
TT_TU     Daily mean air temperature [C]
RF_TU     Daily mean relative humidity [%]
R1        Daily precipitation sum [mm]
SD_SO     Daily sunshine sum
F         Daily mean wind speed [m/s]
P0        Daily mean station pressure [hPa]
```

The dataloader adds cyclical calendar features:

```text
day_sin   Sine encoding of day-of-year seasonality
day_cos   Cosine encoding of day-of-year seasonality
```

Optional rolling/log feature engineering is implemented but disabled by default,
because it did not improve the final WeatherLSTM in the current experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild Processed Data

If the raw DWD directory `XAI_Wetter/` is available locally:

```bash
python build_weather_dataset.py
```

This regenerates the CSV files in `data/processed/`.

## Train

```bash
python train_weather_rnn.py
```

Default training setup:

```text
model: weather_lstm
target: TT_TU
sequence_length: 48
horizon: 1
hidden_size: 64
dropout: 0.15
batch_size: 64
learning_rate: 0.001
patience: 5
device: auto (uses Apple MPS if available, otherwise CUDA, otherwise CPU)
```

Outputs are written to:

```text
outputs/weather_lstm/
```

## Notebook

Open:

```text
notebooks/lstm_training.ipynb
```

It runs the complete workflow:

1. Load and inspect data
2. Build sequence dataloaders
3. Train the WeatherLSTM
4. Compare against Persistence
5. Plot training loss and predictions
