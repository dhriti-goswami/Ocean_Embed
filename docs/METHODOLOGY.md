# Methodology

## 1. Data

| role | variable | product (PS 26066) | native resolution |
|---|---|---|---|
| input | sea surface temperature | OSTIA | 0.05°, daily |
| input | sea surface salinity | SMOS / SMAP | 0.125°, daily |
| input | sea level anomaly (SLA), absolute dynamic topography (ADT) | DUACS | 0.125–0.25°, daily |
| input | surface currents u, v | OSCAR | 0.25°, daily |
| input | 10 m winds u, v | CCMP v3.1 | 0.25°, 6-hourly |
| target | potential temperature `thetao`, 35 levels 0.5–902 m | GLORYS12 reanalysis | 1/12°, daily |
| validation | temperature profiles | ARGO (Ifremer ERDDAP, QC flag 1) | point profiles |

Region: Bay of Bengal, 5–22 °N, 80–100 °E. Period: 1 Jun – 31 Aug 2023 (92 days).

**Harmonization** (team data notebook): conservative/bilinear regridding with xESMF to a
common 0.25° grid (69 × 81), daily averaging of 6-hourly winds, time alignment. The result is
one NetCDF cube; its expected variables are listed in [`data/README.md`](../data/README.md).

**Not used as input:** ASCAT L2 swaths. One swath per day spread by nearest-neighbour
regridding is not a real gridded field, and CCMP already assimilates ASCAT.

## 2. Preprocessing (`oceanembed/data/cube.py`)

- **Masks.** A depth level is ocean at a cell only if the target is valid on every day (the
  seabed cuts columns on the shelf), and a cell is used only if every input is valid on ≥ 90 %
  of days. Loss, metrics and products use only masked cells and levels.
- **Inputs.** 8 satellite fields; SST converted from K to °C. Plus sin/cos of day-of-year,
  latitude and longitude. **Previous 2 days** of the 8 fields are stacked as extra channels
  (28 channels). Only past and present surface data are used, never future days or targets.
- **Split (contiguous in time).** Train 1 Jun – 31 Jul (61 days), validation 1–15 Aug (15),
  test 16–31 Aug (16). Random day-level splits would leak information between near-identical
  neighbouring days.
- **Normalization** per input channel and per depth level, from training days only.
- **Padding** to 72 × 88 so the U-Net and the 8 × 8 ViT patches divide evenly.

## 3. Training (`oceanembed/train/trainer.py`, `configs/main.yaml`)

- Loss = masked MSE on normalized temperature (+ 0.5 × salinity if present)
  \+ 0.1 × Gaussian NLL for the uncertainty head + physics terms (below).
- AdamW (lr 1e-3, weight decay 5e-4), cosine schedule, batch 4, up to 200 epochs, gradient
  clipping at 1.0, input-noise augmentation (σ = 0.05 in normalized units).
- Physics terms are off for 5 warm-up epochs and ramp to full weight over 5 more. The best
  checkpoint (lowest validation RMSE in °C) is only selected **after** the ramp, so a
  "physics" model really was trained with physics. Early stopping after 40 epochs without
  improvement.
- Every configuration is trained with seeds 0, 1, 2.

## 4. Physics-informed losses (`oceanembed/physics/losses.py`)

All terms act on de-normalized predictions and are soft penalties.

**Cyclone-aware ocean-state terms (`phys_mode: cyclone`, main model).** Differentiable
versions of the quantities used for cyclone and ocean forecasting, matched to the same
quantities computed from the target:

| quantity | meaning | soft form (τ = 0.3 °C) | scale |
|---|---|---|---|
| D26 | depth of the 26 °C isotherm | Σ dz · σ((T − 26)/τ) | 10 m |
| TCHP | tropical cyclone heat potential | ρ c_p Σ dz · softplus(T − 26) | 10 kJ/cm² |
| D20 | thermocline depth | Σ dz · σ((T − 20)/τ) | 15 m |
| MLD | mixed layer depth | Σ dz · σ((T − (T₁₀ − 0.5))/0.1) | 5 m |

Loss = w_quant × mean over quantities of ((predicted − target)/scale)².

**Static stability.** Density from a simplified nonlinear equation of state must not decrease
with depth. Without a salinity target the fallback is "temperature must not increase with
depth below 150 m": shallower inversions are real in the Bay of Bengal's fresh upper layer.

**Steric height / geostrophic consistency.** The steric height implied by the predicted
density column (relative to the training-mean density profile) must co-vary spatially with
the observed SLA, which sets the surface geostrophic flow. Penalized only when the prediction
is *less* correlated with SLA than the GLORYS target is.

**v2 constraints mode (`phys_mode: constraints`, ablation).** Mixed-layer homogeneity
instead of the cyclone terms. These inequality-type constraints are already satisfied by a
model fitted to GLORYS, which is why they cannot improve accuracy: a stable-but-wrong profile
costs nothing.

**Not implemented:** mass conservation, which requires a predicted 3D velocity field.

## 5. Uncertainty (`oceanembed/eval/uncertainty.py`)

1. Each seed's head gives σᵢ. Because training residuals are smaller than unseen-day
   residuals, σᵢ is rescaled by one factor kᵢ fitted on the **validation** days
   (kᵢ = RMS of standardized errors).
2. Ensemble of 3 seeds: mean = average of member means; variance = average of kᵢ²σᵢ²
   (what each model does not know) + variance of member means (disagreement).
3. Calibration is checked on **test** days: the share of true values inside the 50–95 %
   intervals, against GLORYS and against ARGO.
4. For ARGO, a float measures a point while GLORYS is a 1/12° model cell. That
   representativeness error (RMS of ARGO − GLORYS per depth band, fitted on non-test days) is
   added in quadrature; results are reported with and without it.

The final product `temperature_std` is the ensemble σ.

## 6. Evaluation

| check | module | what |
|---|---|---|
| held-out days vs GLORYS | `eval/glorys.py` | RMSE, MAE, bias, R² per depth and overall; TCHP, D26, D20, MLD errors; deep-inversion and density-inversion rates |
| independent floats | `eval/argo.py` | model profile interpolated to each float's measurement depths, same cell and day; metrics by depth band; products from ARGO profiles (≥ 10 m to ≥ 200 m) vs model |
| baseline | – | climatology = training-period mean at each cell and depth |
| uncertainty | `eval/uncertainty.py` | coverage vs nominal, by depth band |
| real event | `eval/casestudy.py` | box-mean SST, MLD, D26, TCHP around the 1 Aug 2023 deep depression; ΔTCHP maps |

Product definitions for evaluation are the exact (hard) ones in
`oceanembed/physics/products.py`: linear interpolation of the isotherm crossing, TCHP =
ρ c_p ∫ (T − 26)⁺ dz, MLD by the 0.5 °C criterion relative to 10 m.

## 7. Limitations

- Three months of data (92 days): results describe summer-monsoon conditions only.
- GLORYS assimilates ARGO, so ARGO is independent of the model's inputs and training but not
  perfectly independent of its training target. Test-day ARGO numbers are reported separately.
- ARGO pressure (dbar) is used as depth (m), and in-situ temperature is compared with GLORYS
  potential temperature; both differences are ≲ 0.1 °C above 900 m.
- Overall R² across all depths is inflated by the large surface-to-900 m temperature range;
  per-depth metrics are the meaningful ones.
- Mixed layer depth is not better than climatology, and the post-storm recovery in the case
  study is too fast: 61 training days contain almost no storms.
