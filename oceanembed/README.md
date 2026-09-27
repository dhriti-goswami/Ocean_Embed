# OceanEmbed v2 — satellite-embedding + physics-informed subsurface reconstruction

Reconstructs daily ocean temperature at **35 depth levels (0.5–902 m)** on a **0.25° grid**
over the **Bay of Bengal** from surface satellite observations only, following the
architecture in the SIH 2026 submission (PS 26066).

## Pipeline

```
Data sources (PS-specified)                     Harmonized cube (team data notebook)
  OSTIA SST (0.05°)      SMOS/SMAP SSS (0.125°)   daily, 0.25°, 69 x 81, Jun-Aug 2023
  DUACS SSH/SLA (0.125°) OSCAR currents (0.25°)   xESMF regridding, time harmonization
  CCMP winds (0.25°, 6-hourly -> daily)           target: GLORYS thetao, 35 levels
            |
            v
oceanembed/data.py     masks (land + seabed per level), MLD, contiguous time split,
                       train-only per-channel / per-depth normalization
            |
            v
oceanembed/models.py   ENCODER (satellite embedding)
                         FNO branch  - spectral convolutions (global, large-scale structure)
                         ViT branch  - 8x8 patch tokens + self-attention (long-range links)
                         CNN stem    - local features / skip connections
                         -> fused latent ocean embedding z (128 x H/8 x W/8)
                       DECODER (reconstruction)
                         U-Net upsampling with skips -> T (and S) at every depth
            |
            v
oceanembed/physics.py  PINN losses (soft constraints, ramped in after warm-up)
            |
            v
oceanembed/evaluate.py held-out GLORYS days, per depth, vs climatology + ablations
oceanembed/argo.py     independent ARGO floats (Ifremer ERDDAP, QC=1), by depth band
oceanembed/infer.py    CF NetCDF output (backend-compatible) + embeddings
```

## Physics-informed constraints

| PPT constraint | Implementation | Status |
|---|---|---|
| Mixed-layer-depth consistency | Inside the mixed layer (0.5 °C criterion vs. 10 m, from the target profile) the predicted column must be vertically homogeneous | implemented |
| Thermodynamic stability | Density (simplified nonlinear EOS) must not decrease with depth. Without a salinity target: temperature must not increase with depth **below 150 m** (shallower inversions are real in the Bay of Bengal) | implemented |
| Geostrophic balance | Steric-height consistency: the steric height implied by the predicted density column must co-vary with the observed sea-level anomaly (SSH drives surface geostrophic flow). Penalized only when the prediction is less consistent with observed SSH than the GLORYS target is | implemented (steric-height proxy) |
| Mass conservation | Requires a predicted 3D velocity field, which this model does not output | future work |

If the cube also contains GLORYS salinity (`so`), the model predicts salinity too and the
stability and steric terms use full density; otherwise they use temperature only.

## Experiments reported

| Run | Encoder | Physics | Purpose |
|---|---|---|---|
| `climatology` | – | – | no-skill reference: training-period mean per cell/depth |
| `unet` | CNN only | no | architecture ablation |
| `oceanembed_nophys` | ViT + FNO | no | physics ablation |
| `oceanembed_pinn` | ViT + FNO | yes | **main model** |

Each model is trained with 3 seeds and reported as mean (std), so differences smaller than
run-to-run noise are not over-interpreted. Physics runs: 5 warm-up epochs, 5-epoch ramp, and
checkpoints are only selected once the physics terms are at full weight. All runs use input-noise
augmentation and weight decay (61 training days is small for a 3.6M-parameter network).

Physics diagnostics reported alongside accuracy: deep temperature inversions (> 0.05 °C below
150 m, % of level pairs), density inversions, and mixed-layer-depth RMSE vs. GLORYS.

Split is contiguous in time (train Jun 1–Jul 31, val Aug 1–15, test Aug 16–31); random
day-level splits would leak information between near-identical neighbouring days.

## Run it

Colab (GPU): open `notebooks/run_pipeline_colab.ipynb` and run top to bottom.

Command line:
```bash
python -m oceanembed.train    --data cube.nc --out runs/oceanembed_pinn --variant oceanembed --physics
python -m oceanembed.evaluate --data cube.nc --runs runs/unet runs/oceanembed_nophys runs/oceanembed_pinn --out results
python -m oceanembed.argo     --data cube.nc --runs runs/unet runs/oceanembed_pinn --out results
python -m oceanembed.infer    --data cube.nc --ckpt runs/oceanembed_pinn/best.pt --out outputs
```

Smoke test without real data (synthetic cube, same shapes): `python tests/smoke_test.py`

## Honest notes / limitations

- Three months of data (92 days): results describe summer-monsoon conditions only.
- GLORYS assimilates ARGO, so ARGO is independent of the *model inputs and training* but not
  perfectly independent of the *training target*. Test-period-only ARGO numbers are reported
  separately from all-days numbers.
- ARGO pressure (dbar) is used as depth (m) and in-situ temperature is compared with
  GLORYS potential temperature; both differences are small (≲0.1 °C) above 900 m.
- ASCAT L2 swaths are excluded as inputs: one pass per day regridded by nearest neighbour
  does not form a real gridded field, and CCMP already assimilates ASCAT.
- Overall R² across all depths is inflated by the large surface-to-900 m temperature range;
  per-depth metrics are the meaningful ones.
