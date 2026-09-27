# Architecture

## Pipeline

```
5 surface inputs (SSS, SSH, wind-u, wind-v, bathymetry)
        ↓
Regrid onto common 0.25° grid (ERA5's native resolution)
        ↓
Harmonize time (ERA5 4x/day → daily mean; align all sources' dates)
        ↓
Interpolate GLORYS's 35 native depths → 15 target depths
        ↓
Ocean mask: every input channel AND every target depth valid, every time step
        ↓
CNN (see below)
        ↓
Temperature at 15 standard depths, per grid cell, per day
        ↓
Per-depth model selection (CNN vs. climatological baseline)
```

## Model

**Input:** `(batch, 5, lat, lon)` — 5 stacked surface variable channels
(SSS, SSH, wind-u, wind-v, bathymetry), same idea as R/G/B channels in an
image, just 5 channels of ocean data instead of 3 of color.

**Encoder:** 3 convolutional layers, `kernel_size=3`, `padding=1` (`same`
padding — output spatial size matches input, since we need a full spatial
map back, not a single classification), ReLU activation, 32 hidden
channels throughout.

**Regression head:** a single `1x1` convolution mapping the 32 hidden
channels to 15 output channels (one per target depth). A `1x1` conv acts
as a per-pixel fully-connected layer — it turns the encoder's learned
features into per-pixel temperature predictions at each of the 15 depths,
without mixing information across neighboring pixels at this final step.

**Output:** `(batch, 15, lat, lon)` — predicted temperature at 15 depths,
per grid cell.

Implementation: `src/models/cnn.py::OceanCNN`.

## Why this architecture

Deliberately the simplest CNN that could prove the full data pipeline
works end-to-end (real data in, validated predictions out), rather than
starting with a more complex architecture and risking an unvalidated,
unfinished system. See the main README's "Future Work" section for the
planned next-phase architecture (PINN).

## Loss function

Masked MSE (`src/models/cnn.py::masked_mse_loss`): mean squared error
computed only over cells where the ocean mask is `True`, so land/invalid
cells never contribute to the gradient. The mask is broadcast across all
15 depth channels before applying.
