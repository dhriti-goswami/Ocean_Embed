# OceanEmbed — Subsurface Temperature Reconstruction

CNN that reconstructs a subsurface ocean temperature profile (15 depth
levels, 0-1000m) from surface satellite/reanalysis observations over the
Bay of Bengal — independently validated against real ARGO float
measurements, with a proven, tested bias correction.

Built for Smart India Hackathon 2026 (Problem Statement 26066, Team
BitShifters98).

## Status

✅ **Working, validated baseline.** Trained on 4 months of real data
(June-Sept 2023), evaluated on held-out days, and independently validated
against 162 real ARGO float profiles. All numbers below are real, measured
results — nothing is projected or invented.

## What this does

```
5 surface inputs (SSS, SSH, wind-u, wind-v, bathymetry)
        ↓
CNN (3-layer conv encoder + 1x1 conv regression head)
        ↓
Temperature at 15 standard depths (0, 5, 10, 20, 30, 50, 75, 100,
125, 150, 200, 300, 500, 700, 1000m)
        ↓
Evaluated against held-out GLORYS reanalysis days
        ↓
Independently validated against real ARGO float measurements
```

## Data sources

| Source | Variables | Native resolution | Access |
|---|---|---|---|
| GLORYS12V1 (CMEMS) | temperature (35 native depths), salinity, SSH, currents | 0.083° | `copernicusmarine` Python API |
| ERA5 (CDS) | 10m wind (u, v) | 0.25° | `cdsapi` Python API |
| GEBCO_2023 | bathymetry | ~15 arc-sec, static | GEBCO Grid Subsetting App |
| ARGO floats (Ifremer ERDDAP) | independent temperature measurements | point observations | direct HTTP/CSV, see `src/eval/argo_direct.py` |

All sources regridded onto a common 0.25° grid (ERA5's native resolution).
Region: 80-95°E, 5-22°N. Training window: June 1 - Sept 30, 2023 (122 days,
100 train / 22 validation, split by time — not randomly shuffled, since this
is a time series and random splitting would leak nearby-day information
between train and validation).

## Two real bugs found and fixed

Both were found by inspecting actual output, not assumed — documented here
because they materially changed the results and are worth knowing about if
extending this work.

### Bug 1 — mask built from temperature alone let bad values from other variables through

The ocean/land mask was originally built by checking only whether
*temperature* was valid at a cell. But GLORYS's different variables
(temperature, salinity, SSH, currents) don't share identical missing-data
patterns — a cell could have valid temperature but a gap in salinity for
that same cell/day. This let NaN values leak into cells the mask called
"valid," which propagated into the loss and broke training (loss went to
`NaN` starting from epoch 0).

**Fix:** the mask now requires every input channel *and* every target depth
to be valid, at every time step, before a cell counts as usable
(`src/data/preprocess.py::build_ocean_mask`).

### Bug 2 — single global normalization scale hid errors at low-variance depths

Ocean temperature has very different natural variability by depth in this
region/season — e.g. std ≈ 2.2°C at 100m (thermocline) vs. std ≈ 0.2-0.4°C
near the surface and at 1000m. The target temperatures were originally
normalized using one shared mean/std across all 15 depths combined. Since
the thermocline dominates that shared scale, an error at a low-variance
depth looked numerically tiny to the loss function relative to the same
absolute error at the thermocline — so the model had little pressure to fit
the surface/deep layers precisely, and quietly picked up a systematic
positive bias there instead.

**Fix:** normalize each depth independently
(`src/data/dataset.py::OceanDataset`). Confirmed with real data: before the
fix, several depths lost to a trivial "predict the training mean" baseline;
after the fix, the model beat that baseline at every depth.

## Novel/notable design choices

### Per-depth model selection (hybrid)

Rather than assuming the CNN is the right choice at every depth, each
depth's validation performance is checked against a trivial "predict the
historical mean" baseline (`src/eval/hybrid.py`). The system uses whichever
one genuinely performs better, per depth, decided from validation data
only. In the current trained run, the CNN wins at 14 of 15 depths; one
depth (30m) narrowly favors the baseline. This guarantees the deployed
system is never worse than a trivial constant prediction anywhere, and is
transparent about exactly where the model adds real value (mainly the
thermocline, 50-300m).

### Bias correction, validated on a held-out split

ARGO validation revealed a systematic warm bias at several depths
(up to +1.0°C at 75-100m) — GLORYS-trained predictions running warmer than
real float measurements. A per-depth additive correction was learned from
half of the matched ARGO profiles and tested on the *other*, untouched
half, to avoid circularity. Result: real improvement at every depth on data
the correction never saw (overall bias dropped from +0.287°C to -0.053°C;
R² improved at all 15 depths). See `src/eval/argo_direct.py` and
`checkpoints/bias_correction.npz`.

## Results

### GLORYS-based evaluation (held-out days, same source as training)

| Depth | RMSE (°C) | R² | Correlation |
|---|---|---|---|
| 0m | 0.46 | -0.36 | 0.50 |
| 50m | 0.66 | 0.60 | 0.80 |
| 100m | 0.84 | 0.86 | 0.93 |
| 150m | 0.70 | 0.82 | 0.91 |
| 300m | 0.22 | 0.65 | 0.84 |
| 700m | 0.17 | 0.37 | 0.66 |
| 1000m | 0.23 | -0.07 | 0.52 |

Full per-depth table: `results/metrics/FINAL_results.txt`.

### Independent ARGO validation (real float measurements, 162 matched profiles)

Every depth shows **positive R²** against real, independent physical
sensors — stronger evidence than the GLORYS-only evaluation, since ARGO was
never used in training.

| Depth | RMSE (°C) | R² | Bias |
|---|---|---|---|
| 0m | 0.42 | 0.02 | +0.20 |
| 50m | 1.18 | 0.61 | +0.63 |
| 100m | 1.63 | 0.57 | +1.05 |
| 300m | 0.21 | 0.58 | -0.00 |
| 1000m | 0.16 | 0.24 | -0.08 |

### After bias correction (tested on held-out half of ARGO profiles)

R² improved at every depth; overall bias corrected from +0.287°C to
-0.053°C. E.g. 100m: R² 0.50 → 0.66; 1000m: R² 0.30 → 0.42. Full numbers in
`results/metrics/FINAL_results.txt`.

## Known limitations

- **Single season, 4-month window.** Training data spans June-Sept 2023
  only (monsoon season). Surface and deep layers show low natural
  temperature variability in this specific window, which makes R² an
  unstable metric there even when absolute error (RMSE) is small — this is
  a property of the season/metric, not evidence the model is unusable at
  those depths (confirmed directly: the model beats a trivial baseline at
  14/15 depths). A full-year training window would give more natural
  variation across seasons and likely improve this further.
- **Depth interpolation at the edges.** GLORYS's native depth levels span
  ~0.49m to ~902m; the 0m and 1000m targets required a small linear
  extrapolation for the 4-month training run. Re-pulling with
  `maximum_depth=1100` (as done for ARGO-validation predictions) resolves
  this for future training runs.
- **Systematic warm bias vs. real measurements**, corrected but not
  eliminated at its root — likely reflects GLORYS's own bias relative to
  real sensors in this region/season, or the model learning GLORYS-specific
  patterns rather than true reality. The correction is a calibration
  layer, not a fix to the underlying cause.
- **Baseline architecture**, not the target architecture. See Future Work.

## Future work — Physics-Informed Neural Network (PINN)

The current CNN is a deliberately simple baseline, built first to validate
the full data pipeline (regridding, masking, depth interpolation, ARGO
matching) end-to-end before adding architectural complexity. The planned
next phase replaces/extends the CNN with a PINN — adding loss terms for
known physical constraints (mass conservation, thermodynamic consistency,
geostrophic balance) rather than relying on data patterns alone. This
should help most at the depths where the current model has the least
signal to learn from (very low natural variability), since physical
constraints hold regardless of how much variation exists in a given
season's data. The data pipeline, masking, evaluation, and ARGO-matching
code built here are architecture-agnostic and carry over directly to a PINN
version — only `src/models/` changes.

## Project structure

```
oceanembed/
├── data/                        # local data cache — gitignored, see data/README.md
├── docs/                        # architecture & methodology write-ups
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
├── results/metrics/              # saved numeric results
└── checkpoints/                  # trained weights, normalization stats, bias correction — gitignored except final artifacts
```

## Quick start

```bash
git clone https://github.com/dhriti-goswami/Ocean_Embed.git
cd Ocean_Embed
pip install -r requirements.txt
```

Data must be downloaded separately (see `data/README.md` for exact source
IDs, date ranges, and API calls used). Once downloaded, run the pipeline
via `src/data/preprocess.py::run_pipeline(...)`, train with
`src/models/cnn.py::OceanCNN`, and evaluate with `src/eval/metrics.py` and
`src/eval/argo_direct.py`.

## Requirements

See `requirements.txt`. Python 3.10+ recommended. `argopy` is **not**
required — ARGO data is fetched directly via HTTP (see
`src/eval/argo_direct.py`), since `argopy`/`erddapy` had an unresolved
version conflict in testing.

## Disclaimer

Research/hackathon prototype. Not for operational or safety-critical use.
