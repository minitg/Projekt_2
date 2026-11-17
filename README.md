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

## Precipitation dataset for the multi-model weather browser

The new pipeline builds a separate, auditable dataset from the hourly CSV.
It uses observed precipitation targets and preserves input missingness.
It does not use the globally imputed `data/final.parquet` export.

```bash
python build_precipitation_dataset.py
python evaluate_precipitation_baselines.py
python -m unittest discover -s tests -v
```

Outputs: `data/processed/precipitation_v2/`. Start with its `DATASET.md`,
`split_summary.csv`, and `manifest.json`. This generated directory is ignored
by Git and can be rebuilt from the source CSV and versioned code.

The baseline runner writes `scores.csv`, per-case `predictions.parquet` and the
fitted count tables to `outputs/precipitation_baselines_v2/`. It fits only on
training cases and reports validation and retrospective evaluation separately.
Missing last-observation blocks retain an unknown state. All baselines are
evaluated on the exact same eligible cases within each horizon and split.

Defaults:

- 48 hourly input rows per forecast, with 20 features per hour (six physical
  variables, six missing flags, four carried-value flags, four calendar features).
- UTC-aware timestamps, verified against the original metadata for Regensburg
  and an exact comparison of the CSV with the original hourly products.
- Primary source period starts in 2018, the first complete year after installation
  of the rain[e]H3 gauge (2017-11-14). Earlier measurements remain in the source.
- The reference-to-rain-interval-end offset is estimated at -10 minutes from
  comparison with primary DWD 10-minute data in every year 2015–2024.
- Planned issue times follow the 3-hour reference grid minus 10 minutes:
  23:50, 02:50, 05:50, etc. UTC. `issue_reference_time` and
  `target_reference_end` retain the original label convention; input start/end
  are also source reference timestamps. Physical interval timing is an estimate.
- Latest rain input interval is estimated to end one hour before issuance. This
  is a hypothetical availability scenario. DWD's hourly CDC `recent` archive is
  updated daily through yesterday; finalized historical values are not a live feed.
- Future 3-hour precipitation blocks ending 3, 6, 12 or 24 hours after issuance.
- Binary wet/dry labels and classes `0 mm`, `>0 to <3 mm`, `>=3 mm`.
- All three target measurements must be present and pass basic plausibility
  checks. Missing targets remain unknown and are excluded from model arrays.
- All three target hours must have an explicitly accepted DWD quality procedure
  code (default: QN=3 automatic checking/correction or QN=10 completed checking).
  Observed targets with other codes remain in the audit table but are excluded
  from model arrays. These codes are not a continuous numeric quality scale.
- Training before 2022, validation in 2022–2023, retrospective evaluation in 2024.
  Issuance and all target timestamps must be in the same partition. Earlier
  history is allowed for validation/evaluation predictions.
- Only observed training values determine medians, means and standard deviations.
  Continuous inputs may be carried forward for at most three hours. Rain and
  sunshine sums are never carried forward. Missing flags remain set after fills.

Use the same inputs for both model families:

```python
from precipitation_data import load_precipitation_bundle

data = load_precipitation_bundle("data/processed/precipitation_v2")
x_lstm, y, cases = data.arrays(3, "train")           # (N, 48, 20)
x_xgb, y, cases = data.arrays(3, "train", flatten=True)  # (N, 960)
```

The default builder requires `data/provenance/dwd_04104/audit.json`, which binds
the UTC and source verification to the exact CSV checksum. Original parameter,
instrument and station metadata, source quality flags and interval-comparison
results are saved alongside it. The audit can be reproduced with:

```bash
python audit_dwd_metadata.py --raw-root /path/to/XAI_Wetter \
  --ten-minute-archive data/raw_audit/10minutenwerte_nieder_04104_20100101_20191231_hist.zip \
  --ten-minute-archive data/raw_audit/10minutenwerte_nieder_04104_20200101_20251231_hist.zip
```

The two small 10-minute archives are cached locally and ignored by Git; their
official URLs and checksums are recorded in the audit. Local station metadata is
from DWD, generated 2025-05-15 (CC BY 4.0). The audit's official dataset links
describe the publication cadence and the instrument-specific lag.

`manifest.json` marks the dataset as retrospective and not operational. Neither
historical receipt times nor all point-in-time data revisions are available.
Convert to local display time with `tz_convert("Europe/Berlin")`; never add a
fixed hour offset. The already explored 2024 data is not an untouched final test.
Class thresholds are project definitions, not official weather-warning criteria.
Wind direction remains excluded from the initial feature set. Old v1 outputs
remain available as an earlier exploratory snapshot; use v2 for new experiments.

## Three-class benchmark and foundation-model adapter inputs

The primary task remains the future **three-hour precipitation class**:
`0 = 0 mm`, `1 = >0 to <3 mm`, `2 = >=3 mm`. These are project classes,
not official warning levels. Build an additional classification view from v2:

```bash
python build_precipitation_classification_dataset.py
```

Output: `data/processed/precipitation_classification_v1/`. This keeps all v2
eligible cases, splits and observations unchanged and separates `cases.parquet`
from `labels.parquet`. `schema.json` specifies the target and probability order;
`class_balance.json` reports the unmodified class frequencies per split/horizon.

```python
from classification_data import load_classification_bundle

data = load_classification_bundle("data/processed/precipitation_classification_v1")
x, y, cases = data.arrays(3, "train")                 # (N, 48, 20), classes 0/1/2
x_flat, y, cases = data.arrays(3, "train", flatten=True)  # (N, 960)
record = next(data.foundation_records(3, "validation"))
```

Foundation records contain observed historical rain in mm (missing values stay
NaN), historical covariates in original units plus missing/carried flags, and
only known calendar covariates in the future. The forecast grid includes the
withheld observation gap; `class_target_hour_indices` identifies the three hours
to be classified. Records are model-neutral adapter inputs, not native calls to
Chronos, Timestral or another model library.

Numerical forecasting models still need a trained classification head/adapter
or a justified predictive-distribution conversion to return three probabilities.
Never forecast class IDs as rainfall amounts or infer class probabilities by
thresholding a mean. Summing marginal hourly quantiles is not a joint three-hour
distribution. Training, calibration and model comparison remain subsequent work.

Project rationale and usage: [classification guide](../docs/niederschlag_klassifikation.md).

## TiViT precipitation classification

Start with `notebooks/precipitation_tivit.ipynb`. Install its optional dependencies
with `python -m pip install -r requirements-tivit.txt` in the notebook environment.
It uses the existing audited classification dataset and initially a 3-hour horizon.
The smaller frozen CLIP ViT-B/32 variant is a local TiViT adaptation, not a
reproduction of the paper's ViT-H benchmark. All 20 historical channels are
encoded separately; only the logistic classification head is trained.
Outputs include probabilities with case IDs, metrics, confusion matrices and
run metadata. `MAX_CASES=None` uses all eligible cases; subsets are technical
checks and must not be mixed with full benchmark scores.
