# Training methodology

A practical reference for training and evaluating OceanEmbed: what the model predicts, what
every setting does, how a run proceeds, and the defects found and fixed along the way.
Formal definitions are in [`METHODOLOGY.md`](METHODOLOGY.md), symbols in
[`NOTATION.md`](NOTATION.md).

## Contents

1. [What the model predicts](#1-what-the-model-predicts)
2. [Understanding the metrics](#2-understanding-the-metrics)
3. [Config parameters explained](#3-config-parameters-explained)
4. [Inputs](#4-inputs)
5. [A training run, step by step](#5-a-training-run-step-by-step)
6. [Physics-informed losses in practice](#6-physics-informed-losses-in-practice)
7. [Uncertainty in practice](#7-uncertainty-in-practice)
8. [Defects found and fixed](#8-defects-found-and-fixed)
9. [Reproducibility](#9-reproducibility)

---

## 1. What the model predicts

For every day and every 0.25° ocean cell in the Bay of Bengal, from surface satellite data
only:

- temperature at 35 depth levels, 0.5–902 m;
- a 1-σ uncertainty for each of those values;
- derived products: TCHP, D26, D20, MLD.

It does **not** forecast: it reconstructs the subsurface state on the same day as the
satellite observations (using the previous 2 days as context).

## 2. Understanding the metrics

| metric | meaning | good sign |
|---|---|---|
| `val RMSE` (training log) | temperature RMSE in °C over all valid ocean levels, validation days | falls, then flattens |
| `data` | masked MSE in normalized units (training days) | falls steadily |
| `q_D26`, `q_TCHP`, `q_D20`, `q_MLD` | cyclone-quantity terms, squared errors in units of their scales | fall after physics switches on |
| `stability`, `steric` | constraint terms | near zero is expected (GLORYS-fitted models are already stable) |
| `nll` | uncertainty head likelihood (can be negative) | falls |
| RMSE vs climatology | the main skill number: how much better than the training-period mean | > 0 % improvement |
| per-depth RMSE | where the model is good or bad | largest errors in the thermocline (50–200 m) are normal |
| coverage | share of truths inside the predicted interval | close to nominal (90 % → ~90 %) |

Overall R² across all depths is inflated by the large surface-to-900 m temperature range
(even climatology scores ~0.99). Use RMSE and per-depth numbers.

## 3. Config parameters explained

Set in `configs/*.yaml`; any key can be overridden, e.g. `--epochs 50`.

| key | default (main) | what it does |
|---|---|---|
| `variant` | `oceanembed` | `oceanembed` = FNO + ViT + CNN stem; `unet` = plain U-Net (ablation) |
| `uncertainty` | `true` | adds the log-variance head and NLL loss |
| `history` | `2` | previous days of surface data as extra input channels |
| `physics` | `true` | enables physics terms |
| `phys_mode` | `cyclone` | `cyclone` = TCHP/D26/D20/MLD + stability + steric; `constraints` = v2 terms |
| `w_quant` | 0.1 | weight of the cyclone-quantity terms |
| `w_stab` | 1.0 | weight of static stability |
| `w_steric` | 0.1 | weight of steric-height/SLA consistency |
| `w_mld` | 0.05 | weight of MLD homogeneity (constraints mode only) |
| `warmup` | 5 | epochs of data-only training before physics |
| `ramp` | 5 | epochs to ramp physics from 0 to full weight |
| `epochs` | 200 | maximum epochs |
| `batch` | 4 | days per batch |
| `lr` | 1e-3 | AdamW learning rate (cosine decay) |
| `wd` | 5e-4 | weight decay |
| `noise` | 0.05 | Gaussian input noise on ocean cells (augmentation) |
| `patience` | 40 | early stopping patience (epochs) |
| `seed` | 42 | random seed; reported results use 0, 1, 2 |

## 4. Inputs

28 channels: SST, SSS, SLA, ADT, OSCAR u/v, CCMP u/v at days t−2, t−1, t (24), plus
sin/cos day-of-year, latitude, longitude (4). Why the previous days: the subsurface responds
to surface forcing with a lag (wind mixing deepens the mixed layer over days; SSH changes
track thermocline displacement). In the ablations this lowered RMSE by ~5 %.

## 5. A training run, step by step

1. Load the cube, build masks, split by time, normalize with training-day statistics.
2. Epochs 1–5: data loss (+ NLL) only.
3. Epochs 6–10: physics terms ramp up linearly.
4. From epoch 10: full loss. After each epoch, validation RMSE (°C) is computed; the best
   checkpoint is saved **only from this point on**, so a physics run's checkpoint was really
   trained with physics.
5. Stop when validation RMSE has not improved for 40 epochs (or at 200).
6. Outputs: `best.pt` (weights + config + normalization), `history.json`, `summary.json`.

On a Colab T4 one run takes about a minute; the full study (4 configs × 3 seeds + all
evaluation) about 20–25 minutes.

## 6. Physics-informed losses in practice

- **Constraint terms stay near zero.** A model fitted to GLORYS already produces stable
  columns (0 % deep inversions without any physics), so stability and steric terms rarely
  activate. They are kept as safeguards, not as accuracy levers.
- **Cyclone-quantity terms are active.** Their values fall steadily after the ramp; they are
  the only physics terms that pull a profile toward the right structure.
- **Measured effect:** best D26/TCHP errors vs GLORYS and the smallest seed-to-seed spread,
  but gains over the no-physics model are within seed variability. See
  [`RESULTS.md`](RESULTS.md).

## 7. Uncertainty in practice

- The head is trained on training days, where residuals are small; raw σ is therefore too
  small on unseen days. Each seed is rescaled with one factor fitted on validation days.
- Three seeds are combined; their disagreement is added to the variance.
- Against ARGO, point-vs-model-cell mismatch is added (`σ_repr`). Without it, coverage against
  floats is far too low; with it, slightly conservative. Both are reported.

## 8. Defects found and fixed

Recorded so that nobody reintroduces them.

| # | defect | symptom | fix |
|---|---|---|---|
| 1 | v1 never used SST, and its "surface" inputs were GLORYS fields rather than satellite products | near-zero or negative R² at 0–30 m in v1; mismatch with PS 26066 | v2 inputs replaced with OSTIA, SMOS/SMAP, DUACS, OSCAR, CCMP |
| 2 | ASCAT L2 swaths regridded by nearest neighbour spread a single pass over the whole grid | repeated identical values (e.g. `1.8 1.8 1.8 …`) in the cube | ASCAT channels excluded (CCMP already assimilates ASCAT); `test_ascat_swath_channels_are_not_inputs` |
| 3 | physics warm-up/ramp overlapped with checkpoint selection | "PINN" checkpoint saved at epoch 21 with ~5 % physics weight: effectively a no-physics model | checkpoints selected only after the ramp completes |
| 4 | stability metric did not match the stability loss | 14 % "unstable pairs" reported, mostly sub-0.05 °C wiggles between closely spaced shallow levels, with salinity fixed at 35 | added `deep_T_inv_pct` (same definition as the loss); both reported |
| 5 | inequality constraints cannot improve a model fitted to a physically consistent target | PINN = no-physics on every metric | added cyclone-aware quantity losses (`phys_mode: cyclone`) |
| 6 | 3.6 M parameters, 61 training days | best validation epoch ≈ 20, then rising error | input-noise augmentation, weight decay 5e-4 |
| 7 | uncertainty calibrated against GLORYS ignored float-vs-cell mismatch | on a synthetic test, only 22 % of ARGO values inside the 90 % interval | representativeness term, fitted on non-test days, reported alongside the raw version |
| 8 | `@torch.no_grad()` decorator displaced onto a helper during a refactor | `RuntimeError: Can't call numpy() on Tensor that requires grad` | decorator restored on `predict_all`; covered by the end-to-end test |
| 9 | writing results straight to a Colab-mounted Drive | files missing after the session ended | write locally, copy to Drive after each stage; zip download at the end |

## 9. Reproducibility

- Configs: `configs/*.yaml`; seeds 0, 1, 2.
- Split and normalization are deterministic given the cube.
- GPU nondeterminism (cuDNN, FFT) means reruns differ slightly; that is why results are
  reported as mean ± std over seeds.
- Full study: `make experiments DATA=path/to/cube.nc` or the Colab notebook.
- Tests: `make test` (unit), `make test-all` (including end-to-end).
