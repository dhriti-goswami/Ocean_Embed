# Training Methodology

## Data

Region: 80–95°E, 5–22°N (Bay of Bengal). Training window: June 1 – Sept 30,
2023 (122 days). Sources: GLORYS12V1 (temperature, salinity, SSH,
currents), ERA5 (wind), GEBCO_2023 (bathymetry). See main README's "Data
Sources" table and `data/README.md` for exact API calls and date ranges
used.

## Train / validation split

**Split by time, not randomly.** First 100 days used for training, last
22 days held out for validation. Random shuffling was deliberately avoided
because this is a time series — nearby days are physically similar, so a
random split would let validation days leak information into training
(the model could partly "cheat" by learning from a training day almost
identical to a nearby validation day), inflating apparent performance.

## Normalization

Z-score normalization (`(value - mean) / std`), computed **only on
training data**, then reused (not recomputed) for the validation set — this
avoids validation data influencing the normalization stats, a form of data
leakage.

**Per-depth, not global.** Each of the 15 target depths is normalized
using its own mean and std, not one shared scale across all depths. See
"Bug 2" below for why this matters.

## Loss function and optimizer

Masked MSE loss (see `docs/ARCHITECTURE.md`). Adam optimizer,
learning rate `1e-3`. Batch size 8. 80 epochs (chosen by observing the
training loss curve plateau; no formal early-stopping implemented yet).

## Two real bugs found during training, and their fixes

Both were found by inspecting actual output during this project's
development, not assumed in advance.

### Bug 1 — ocean mask built from temperature alone let bad values from other variables through

**Symptom:** training loss went to `NaN` starting from epoch 0.

**Cause:** the ocean/land mask was originally built by checking only
whether *temperature* was valid (non-NaN) at a given cell. But GLORYS's
different variables (temperature, salinity, SSH, currents) do not
necessarily share identical missing-data patterns — a cell could have
valid temperature down to 1000m but a gap in salinity or current data for
that same cell/day. This let NaN values leak into cells the mask called
"valid," and any arithmetic touching a NaN (loss computation, gradient)
turns the entire result to NaN.

**Fix:** `build_ocean_mask()` now requires every input channel *and* every
target depth to be valid, at every time step, before a cell counts as
usable (`src/data/preprocess.py`).

### Bug 2 — single global normalization scale hid errors at low-variance depths

**Symptom:** on real 4-month training data, several depths (0–30m,
500–1000m) showed model RMSE *worse* than a trivial "always predict the
training-set mean" baseline — confirmed by direct comparison, not assumed.

**Cause:** ocean temperature has very different natural variability by
depth in this region/season — measured directly on real validation data:
std ≈ 0.33–0.39°C at 0–30m, std ≈ 1.7–2.2°C at 75–150m (thermocline), std
≈ 0.22–0.23°C at 500–1000m. Normalizing all 15 depths with one shared
mean/std meant the thermocline (with the largest natural spread) dominated
that shared scale. An error of a given size at a low-variance depth looked
numerically tiny to the loss function relative to the same absolute error
at the thermocline — so the model had little pressure to fit the
surface/deep layers precisely, and picked up a systematic positive bias
there instead.

**Fix:** normalize each depth independently
(`src/data/dataset.py::OceanDataset`). Confirmed with real data: before the
fix, several depths lost to the naive baseline; after the fix, the model
beat the naive baseline at every depth (see `docs/VALIDATION.md` for the
before/after numbers).

## Checkpointing

Final trained weights saved to `checkpoints/oceancnn_4months.pth`,
alongside normalization stats (`checkpoints/normalization_stats.npz`) and
the per-depth model-selection choice
(`checkpoints/hybrid_choice.json`) — all three are required together to
use the trained model correctly (raw inputs must be normalized with the
same stats used in training, and predictions must be denormalized before
interpreting as real temperatures).
