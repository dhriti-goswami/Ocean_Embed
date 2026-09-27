# Architecture

Code: [`oceanembed/models/network.py`](../oceanembed/models/network.py).

## Input and output

| tensor | shape | content |
|---|---|---|
| input `x` | (B, 28, 72, 88) | 8 satellite fields × 3 days (t−2, t−1, t) + sin/cos day-of-year + lat + lon, normalized; 69 × 81 grid zero-padded to 72 × 88 |
| output mean | (B, 35, 72, 88) | normalized temperature at 35 GLORYS levels (0.5–902 m) |
| output log-variance | (B, 35, 72, 88) | per-point temperature uncertainty (if `uncertainty: true`) |
| embedding `z` | (B, 128, 9, 11) | latent ocean state; each cell covers an 8 × 8 block of the 0.25° grid |

If the data cube contains GLORYS salinity (`so`), 35 salinity channels are added to the mean
output and the density-based physics terms use them.

## Network

```
x (28 ch, 72x88)
 |-- FNO branch     lift 1x1 -> 4 x [spectral conv (12x12 modes) + 1x1 conv, GELU]   -> 32 ch, 72x88
 |-- ViT branch     8x8 patch embedding -> 9x11 tokens, learned position embedding,
 |                  4 pre-norm transformer layers (dim 128, 4 heads)                -> 128 ch, 9x11
 |-- CNN stem       conv blocks at 72x88 (32), 36x44 (64), 18x22 (128), 9x11 (128)
 v
fusion at 9x11:  concat[stem 128, ViT 128, avg-pooled FNO 32] -> conv block -> z (128 ch)
 v
U-Net decoder:   up + skip 18x22 (128) -> up + skip 36x44 (64) -> up + skip 72x88 (+FNO) (64)
 v
1x1 head -> 35 mean channels (+35 log-variance channels)
```

Conv block = 2 × (3 × 3 conv, GroupNorm(8), GELU). Dropout2d(0.1) after fusion.

## Parameters (28 input channels, uncertainty head on)

| component | parameters |
|---|---|
| FNO branch | 1.18 M |
| ViT branch | 0.77 M |
| CNN stem (4 blocks) | 0.59 M |
| fusion | 0.48 M |
| U-Net decoder + head | 0.71 M |
| **OceanEmbed total** | **3.73 M** |
| U-Net ablation (`variant: unet`, no FNO/ViT) | 1.57 M |

## Why each branch

- **FNO:** learns in Fourier space, so a few layers see basin-scale patterns (large eddies,
  monsoon-driven gradients) that local convolutions need many layers to reach.
- **ViT:** self-attention links any two regions directly, e.g. a river-plume salinity signal in
  the north and thermocline structure further south.
- **CNN stem + U-Net decoder:** local detail and sharp fronts at full 0.25° resolution, via skip
  connections.

In the ablations the ViT + FNO encoder is slightly more accurate than a plain U-Net and much
more consistent across random seeds (see [RESULTS.md](RESULTS.md)).

## Uncertainty head

The head predicts log σ² per temperature channel. It is trained with a Gaussian negative
log-likelihood on a **detached** mean, so learning the uncertainty cannot degrade the mean
prediction. See [METHODOLOGY.md](METHODOLOGY.md#uncertainty) for calibration and ensembling.
