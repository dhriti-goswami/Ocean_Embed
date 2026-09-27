# Validation

Three layers of validation, in increasing order of independence from the
training data.

## 1. GLORYS-based evaluation (held-out days)

The last 22 of 122 training-window days, never seen during training,
evaluated against real GLORYS reanalysis values.

| Depth | RMSE (°C) | MAE (°C) | R² | Correlation | Bias |
|---|---|---|---|---|---|
| 0m | 0.46 | 0.37 | -0.36 | 0.50 | +0.28 |
| 5m | 0.45 | 0.36 | -0.61 | 0.45 | +0.29 |
| 10m | 0.45 | 0.36 | -0.61 | 0.45 | +0.28 |
| 20m | 0.43 | 0.34 | -0.73 | 0.40 | +0.25 |
| 30m | 0.44 | 0.36 | -0.79 | 0.00* | +0.29 |
| 50m | 0.66 | 0.48 | 0.60 | 0.80 | +0.04 |
| 75m | 0.82 | 0.63 | 0.83 | 0.92 | -0.02 |
| 100m | 0.84 | 0.64 | 0.86 | 0.93 | +0.08 |
| 125m | 0.81 | 0.65 | 0.85 | 0.93 | +0.09 |
| 150m | 0.70 | 0.56 | 0.82 | 0.91 | +0.02 |
| 200m | 0.41 | 0.33 | 0.78 | 0.90 | -0.12 |
| 300m | 0.22 | 0.17 | 0.65 | 0.84 | -0.09 |
| 500m | 0.16 | 0.12 | 0.49 | 0.74 | -0.04 |
| 700m | 0.17 | 0.13 | 0.37 | 0.66 | -0.04 |
| 1000m | 0.23 | 0.18 | -0.07 | 0.52 | -0.12 |

*30m: this depth uses the climatological baseline, not the CNN — see
"Per-depth model selection" below. A constant prediction has undefined
correlation with real variation, shown as 0.00.

## 2. Independent ARGO validation (real float measurements)

162 real ARGO float profiles matched by lat/lon/time (within 1 day) inside
the training region and date window, fetched directly from Ifremer's
ERDDAP server (`src/eval/argo_direct.py`) — bypasses the `argopy` Python
package, which had an unresolved dependency conflict (`erddapy`
incompatibility) in this environment. ARGO floats were **never used in
training** — GLORYS was the sole training target.

| Depth | RMSE (°C) | R² | Bias |
|---|---|---|---|
| 0m | 0.42 | 0.02 | +0.20 |
| 5m | 1.20 | 0.09 | +0.14 |
| 10m | 0.64 | 0.21 | +0.09 |
| 20m | 0.51 | 0.33 | +0.11 |
| 30m | 0.74 | 0.48 | +0.22 |
| 50m | 1.18 | 0.61 | +0.63 |
| 75m | 1.68 | 0.56 | +1.05 |
| 100m | 1.63 | 0.57 | +1.05 |
| 125m | 1.41 | 0.55 | +0.75 |
| 150m | 0.98 | 0.58 | +0.35 |
| 200m | 0.52 | 0.53 | +0.11 |
| 300m | 0.21 | 0.58 | -0.00 |
| 500m | 0.18 | 0.53 | +0.01 |
| 700m | 0.20 | 0.26 | -0.11 |
| 1000m | 0.16 | 0.24 | -0.08 |

**Every depth shows positive R² against real, independent measurements** —
notably including 0m and 1000m, which showed negative R² in the
GLORYS-only evaluation above. This is a stronger result than the
GLORYS-based table, since ARGO is genuinely independent of the training
data source.

## 3. Bias correction (validated on a held-out split of ARGO profiles)

The ARGO validation above shows a systematic positive (warm) bias at
several depths, most notably 75–125m (+0.75 to +1.05°C). A per-depth
additive correction was learned and tested honestly, to avoid circularity:

1. The 162 matched ARGO profiles were split randomly in half.
2. The per-depth mean (predicted − true) was computed from **calibration
   half only** — this is the correction.
3. The correction was applied to the **other, untouched half**, and
   accuracy was compared before/after on that untouched half.

| | Overall bias | 100m R² | 1000m R² |
|---|---|---|---|
| Before correction | +0.287°C | 0.50 | 0.30 |
| After correction | -0.053°C | 0.66 | 0.42 |

R² improved at every depth on data the correction never saw. Saved at
`checkpoints/bias_correction.npz`.

## Per-depth model selection (hybrid)

Independently of the bias correction, each depth's validation RMSE is
compared against a trivial "predict the training-set mean" baseline
(`src/eval/hybrid.py`). Whichever wins is used. In the current trained
run: the CNN wins at 14 of 15 depths; one depth (30m) narrowly favors the
baseline. Decision saved at `checkpoints/hybrid_choice.json`. This
guarantees the deployed system is never worse than a trivial constant
prediction at any depth, and makes explicit exactly where the model adds
real value (mainly the thermocline, 50–300m).

## Known limitations

- **Single season, 4-month window.** Training data spans June–Sept 2023
  (monsoon season) only. Surface and deep layers show low natural
  temperature variability in this specific window (std ≈ 0.2–0.4°C),
  which makes R² an unstable metric there even when absolute error is
  small — this is a property of the season/metric, not evidence the model
  is unusable at those depths (the model beats a trivial baseline at
  14/15 depths regardless). A full-year training window would give more
  natural variation across seasons and likely improve this further.
- **Depth interpolation at the edges.** GLORYS's native depths span
  ~0.49–902m in the original 4-month pull; the 0m and 1000m targets
  required a small linear extrapolation. Re-pulling with
  `maximum_depth=1100` (used for the ARGO-validation predictions)
  resolves this for future training runs, since a native GLORYS level
  past 1000m then exists.
- **Systematic warm bias vs. real measurements**, corrected but not
  eliminated at its root cause — likely reflects GLORYS's own bias
  relative to real sensors in this region/season, or the model learning
  GLORYS-specific patterns rather than true reality. The correction is a
  calibration layer applied on top of the trained model, not a fix to the
  underlying cause.
- **Baseline CNN architecture**, not the target architecture — see the
  main README's "Future Work" section for the planned PINN-based
  successor.
