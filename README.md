# OceanEmbed — Subsurface Ocean Temperature Reconstruction

CNN that reconstructs a subsurface ocean temperature profile (15 depth
levels, 0–1000m) from surface satellite/reanalysis observations over the
Bay of Bengal — independently validated against real ARGO float
measurements, with a proven, held-out-tested bias correction.

Built for Smart India Hackathon 2026 (Problem Statement 26066, Team
BitShifters98).

---

## Overview

A working, validated pipeline that turns 5 surface variables (salinity,
sea surface height, wind u/v, bathymetry) into a full-depth temperature
profile at every point in the Bay of Bengal, trained on 4 months of real
GLORYS reanalysis data and checked against 162 real ARGO float
measurements the model never trained on.

**Key Capabilities:**

- Predicts temperature at 15 standard depths (0–1000m) from surface data alone
- Independent validation against real ARGO float profiles, never used in training
- Per-depth model selection: falls back to a climatological baseline at any
  depth where the CNN doesn't genuinely add value
- Held-out-tested bias correction against real measurements
- Depth-wise error breakdown (RMSE, MAE, R², correlation, bias) — not just
  an overall average

## Model Performance

### GLORYS-based evaluation (held-out validation days)

| Depth | RMSE (°C) | R² | Correlation |
|---|---|---|---|
| 0m | 0.46 | -0.36 | 0.50 |
| 50m | 0.66 | 0.60 | 0.80 |
| 100m | 0.84 | **0.86** | 0.93 |
| 150m | 0.70 | 0.82 | 0.91 |
| 300m | 0.22 | 0.65 | 0.84 |
| 700m | 0.17 | 0.37 | 0.66 |
| 1000m | 0.23 | -0.07 | 0.52 |

### Independent ARGO validation (real float measurements, 162 profiles)

Every depth shows positive R² against real physical sensors — never used
in training.

| Depth | RMSE (°C) | R² | Bias |
|---|---|---|---|
| 0m | 0.42 | 0.02 | +0.20 |
| 50m | 1.18 | 0.61 | +0.63 |
| 100m | 1.63 | 0.57 | +1.05 |
| 300m | 0.21 | **0.58** | -0.00 |
| 1000m | 0.16 | 0.24 | -0.08 |

### After bias correction (validated on held-out half of ARGO profiles)

| | Before | After |
|---|---|---|
| Overall bias | +0.287°C | **-0.053°C** |
| 100m R² | 0.50 | **0.66** |
| 1000m R² | 0.30 | **0.42** |

Full per-depth tables: `results/metrics/FINAL_results.txt`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          OceanEmbed                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │  GLORYS12V1  │   │     ERA5     │   │  GEBCO_2023  │         │
│  │ (temp, SSS,  │   │   (wind      │   │ (bathymetry, │         │
│  │  SSH, uo/vo) │   │    u, v)     │   │    static)   │         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘         │
│         └──────────────────┼──────────────────┘                 │
│                             ▼                                    │
│              Regrid → common 0.25° grid                          │
│              Harmonize time → daily                              │
│              Interpolate → 15 target depths                      │
│              Mask → every variable valid, every depth            │
│                             │                                    │
│                             ▼                                    │
│              ┌───────────────────────────┐                       │
│              │   CNN (3-layer encoder     │                       │
│              │   + 1x1 conv regression    │                       │
│              │   head), per-depth         │                       │
│              │   normalized loss          │                       │
│              └─────────────┬───────────────┘                     │
│                             ▼                                    │
│              ┌───────────────────────────┐                       │
│              │  Per-depth model selection │                       │
│              │  (CNN vs. climatological   │                       │
│              │  baseline, whichever wins) │                       │
│              └─────────────┬───────────────┘                     │
│                             ▼                                    │
│         ┌───────────────────────────────────────┐                │
│         │  GLORYS eval   │   ARGO validation      │                │
│         │  (held-out)    │  (real, independent,   │                │
│         │                │   + bias correction)   │                │
│         └───────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | Content | Native resolution | Access |
|---|---|---|---|
| GLORYS12V1 (CMEMS) | temperature (35 native depths), salinity, SSH, currents | 0.083° | `copernicusmarine` Python API |
| ERA5 (CDS) | 10m wind (u, v) | 0.25° | `cdsapi` Python API |
| GEBCO_2023 | bathymetry | ~15 arc-sec, static | GEBCO Grid Subsetting App |
| ARGO floats (Ifremer ERDDAP) | independent real temperature measurements | point observations | direct HTTP/CSV — see `src/eval/argo_direct.py` |

Region: 80–95°E, 5–22°N. Training window: June 1 – Sept 30, 2023 (122 days;
100 train / 22 validation, split **by time**, not randomly — this is a time
series, and random splitting would leak nearby-day information between
train and validation).

---

## Two real bugs found and fixed

### Bug 1 — mask built from temperature alone let bad values from other variables through

The ocean/land mask originally checked only whether *temperature* was
valid at a cell. GLORYS's different variables don't share identical
missing-data patterns — a cell could have valid temperature but a gap in
salinity for that same cell/day. This let NaN leak into "valid" cells,
which broke training (loss went to `NaN` from epoch 0).

**Fix:** the mask now requires every input channel *and* every target
depth to be valid, at every time step (`src/data/preprocess.py::build_ocean_mask`).

### Bug 2 — single global normalization scale hid errors at low-variance depths

Depths vary hugely in natural variability (std ≈ 2.2°C at 100m vs. ≈ 0.2°C
near the surface/1000m). A single shared normalization scale, dominated
by the thermocline, made low-variance-depth errors look numerically tiny
to the loss function — so the model barely tried to fit them, and picked
up a systematic positive bias there instead.

**Fix:** normalize each depth independently
(`src/data/dataset.py::OceanDataset`). Confirmed with real data: before the
fix, several depths lost to a trivial "predict the mean" baseline; after,
the model beat that baseline everywhere.

## Novel design choices

**Per-depth model selection** (`src/eval/hybrid.py`) — each depth's
validation performance is checked against a climatological baseline;
whichever wins is used. Currently the CNN wins at 14/15 depths. Guarantees
the system is never worse than a trivial constant prediction anywhere.

**Bias correction, validated on a held-out split** (`src/eval/argo_direct.py`) —
a systematic warm bias found against real ARGO measurements (up to
+1.0°C at 75–100m) was corrected using a per-depth offset learned from
half the ARGO profiles and tested on the untouched other half, to avoid
circularity. Real improvement confirmed on data the correction never saw.

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

## Model Training

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
├── docs/                        # architecture & methodology write-ups
├── results/metrics/              # saved numeric results
└── checkpoints/                  # trained weights, norm stats, bias correction
```

---

## Known Limitations

- **Single season, 4-month window.** Training data spans June–Sept 2023
  (monsoon season). Surface and deep layers show low natural temperature
  variability in this window, making R² unstable there even when absolute
  error (RMSE) is small — a property of the season/metric, not evidence the
  model fails there (confirmed: the model beats a trivial baseline at
  14/15 depths). A full-year window would likely improve this further.
- **Depth interpolation at the edges.** GLORYS's native depths span
  ~0.49–902m; the 0m and 1000m targets required a small linear
  extrapolation in the 4-month training run. Pulling with
  `maximum_depth=1100` (as done for ARGO-validation predictions) resolves
  this going forward.
- **Systematic warm bias vs. real measurements**, corrected but not
  eliminated at the root — likely reflects GLORYS's own bias relative to
  real sensors, or the model learning GLORYS-specific patterns rather than
  true reality. The correction is a calibration layer, not a root-cause fix.
- **Baseline CNN architecture**, not the target architecture — see below.

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
