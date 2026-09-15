# OceanEmbed — Subsurface Ocean Temperature Reconstruction

CNN that reconstructs a subsurface ocean temperature profile (15 depth
levels, 0–1000m) from surface satellite/reanalysis observations over the
Bay of Bengal — independently validated against real ARGO float
measurements, with a held-out-tested bias correction.

Built for Smart India Hackathon 2026 (Problem Statement 26066, Team
BitShifters98).

---

## Overview

A working, validated pipeline that turns 5 surface variables (salinity,
sea surface height, wind u/v, bathymetry) into a full-depth temperature
profile at every point in the Bay of Bengal, trained on 4 months of real
GLORYS reanalysis data and independently checked against 162 real ARGO
float measurements the model never trained on.

**Key Capabilities:**

- Predicts temperature at 15 standard depths (0–1000m) from surface data alone
- Independent validation against real ARGO float profiles, never used in training
- Per-depth model selection: falls back to a climatological baseline at any
  depth where the CNN doesn't genuinely add value
- Held-out-tested bias correction against real measurements
- Depth-wise error breakdown (RMSE, MAE, R², correlation, bias) — not just
  an overall average

## Model Performance

| Validation | Overall RMSE | Overall R² | Notes |
|---|---|---|---|
| GLORYS (held-out days) | 0.53°C | 0.996 | Same source as training; 3 depths (0-30m, 1000m) show low R² due to low natural variability that season — see `docs/VALIDATION.md` |
| ARGO (real, independent) | 0.96°C | 0.987 | **Every depth positive R²**, including the ones that struggled against GLORYS |
| ARGO, after bias correction (held-out test half) | 0.76°C | 0.992 | Bias corrected from +0.287°C → -0.053°C; RMSE improved from 0.87°C, tested on data the correction never saw |

Full per-depth tables, exact methodology, and limitations:
[`docs/VALIDATION.md`](docs/VALIDATION.md).

---

## Quick Start

```bash
git clone https://github.com/dhriti-goswami/Ocean_Embed.git
cd Ocean_Embed
pip install -r requirements.txt
```

Data must be downloaded separately — see `data/README.md` for the exact
source IDs, date ranges, and API calls used.

## Usage

### Run the full pipeline

```python
from src.data.preprocess import run_pipeline, TARGET_DEPTHS
from src.data.dataset import OceanDataset
from src.models.cnn import OceanCNN, masked_mse_loss

X, Y, ocean_mask, channel_names, common_times = run_pipeline(
    "glorys_temp.nc", "glorys_surface.nc", "era5_wind.nc", "gebco.nc"
)
```

### Train

```python
import torch
from torch.utils.data import DataLoader

train_ds = OceanDataset(X[:100], Y[:100], ocean_mask)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = OceanCNN(in_channels=5, out_depths=15)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(80):
    for xb, yb, mb in train_loader:
        optimizer.zero_grad()
        loss = masked_mse_loss(model(xb), yb, mb)
        loss.backward()
        optimizer.step()
```

### Evaluate against held-out days

```python
from src.eval.metrics import compute_metrics, print_metrics_report
metrics = compute_metrics(val_true, val_pred, ocean_mask)
print_metrics_report(metrics, TARGET_DEPTHS)
```

### Validate against real ARGO floats

```python
from src.eval.argo_direct import fetch_argo_csv, match_argo_to_predictions

argo_df = fetch_argo_csv(lon_min=80, lon_max=95, lat_min=5, lat_max=22,
                          date_min="2023-06-01", date_max="2023-09-30")
matched_true, matched_pred = match_argo_to_predictions(
    argo_df, predict_fn, model_lat, model_lon, model_times, TARGET_DEPTHS
)
```

**Training output:** model checkpoint (`checkpoints/oceancnn_4months.pth`),
normalization stats (`checkpoints/normalization_stats.npz`), bias
correction (`checkpoints/bias_correction.npz`).

## Project Structure

```
Ocean_Embed/
├── src/
│   ├── data/
│   │   ├── preprocess.py        # regrid, harmonize time, depth-interpolate, mask
│   │   └── dataset.py           # PyTorch Dataset, per-depth normalization
│   ├── models/
│   │   └── cnn.py               # baseline CNN + masked loss
│   └── eval/
│       ├── metrics.py           # RMSE/MAE/R²/correlation/bias, overall + per-depth
│       ├── hybrid.py            # per-depth CNN-vs-baseline model selection
│       └── argo_direct.py       # direct ARGO fetch (Ifremer ERDDAP) + matching
├── data/                        # local data cache — gitignored, see data/README.md
├── docs/
│   ├── ARCHITECTURE.md          # model design, layer-by-layer
│   ├── TRAINING_METHODOLOGY.md  # split strategy, normalization, both bugs found & fixed
│   └── VALIDATION.md            # full results tables, bias correction, limitations
├── results/metrics/              # saved numeric results
└── checkpoints/                  # trained weights, norm stats, bias correction
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — model design, input/output shapes, layer-by-layer
- [`docs/TRAINING_METHODOLOGY.md`](docs/TRAINING_METHODOLOGY.md) — split strategy, normalization approach, both real bugs found and fixed during development
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — full per-depth results (GLORYS + ARGO + bias-corrected), the hybrid model-selection mechanism, and known limitations

## Future Work — Physics-Informed Neural Network (PINN)

The current CNN is a deliberately simple baseline, built first to validate
the full data pipeline end-to-end before adding architectural complexity.
The planned next phase adds a PINN — loss terms for known physical
constraints (mass conservation, thermodynamic consistency, geostrophic
balance) rather than relying on data patterns alone. This should help most
at depths where the current model has the least natural signal to learn
from, since physical constraints hold regardless of a given season's data
variability. The data pipeline, masking, evaluation, and ARGO-matching
code here are architecture-agnostic and carry over directly — only
`src/models/` changes.

## Requirements

Python 3.10+, PyTorch 2.0+. See `requirements.txt`. `argopy` is **not**
required — ARGO data is fetched directly via HTTP
(`src/eval/argo_direct.py`), since `argopy`/`erddapy` had an unresolved
version conflict in testing.

## Disclaimer

Research/hackathon prototype. Not for operational or safety-critical use.
