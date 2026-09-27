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



## Results (Bay of Bengal, 1 Jun – 31 Aug 2023)

Train 1 Jun–31 Jul, validation 1–15 Aug, **test 16–31 Aug** (never used for training or model
selection). Every model trained with 3 seeds; values are mean ± std across seeds.
Baseline "climatology" = training-period mean at each cell and depth.

### Temperature, 0–902 m, held-out test days vs GLORYS

| model | RMSE (°C) | bias (°C) |
|---|---|---|
| climatology | 0.974 | +0.318 |
| U-Net | 0.653 ± 0.052 | +0.101 ± 0.151 |
| ViT+FNO, no physics | 0.630 ± 0.008 | −0.108 ± 0.018 |
| ViT+FNO, v2 constraints | 0.629 ± 0.018 | −0.120 ± 0.027 |
| **ViT+FNO, cyclone-aware (main)** | **0.627 ± 0.011** | −0.104 ± 0.008 |
| ViT+FNO + v2 constraints, *without* previous-day inputs | 0.661 ± 0.016 | −0.009 ± 0.021 |

The main model's error is **~35 % below climatology**. Adding the previous 2 days of surface data
reduced error by ~5 % (0.661 → 0.63). The ViT+FNO encoder is slightly better and much more
consistent across seeds than a plain U-Net. Physics losses give no significant change in overall
temperature RMSE. Deep (>150 m) temperature inversions: < 0.01 % for all models.

### Cyclone-relevant products, test days

| product | climatology vs GLORYS | main model vs GLORYS | climatology vs **ARGO** | main model vs **ARGO** |
|---|---|---|---|---|
| TCHP (kJ/cm²) | 21.5 | **13.2 ± 0.4** | 29.8 | **16.5** |
| D26 (m) | 15.0 | **8.4 ± 0.1** | 22.0 | 13.4 |
| D20 (m) | 17.1 | 9.2 ± 0.2 | 23.3 | 14.9 |
| MLD (m) | 12.3 | 12.0 ± 0.6 | 16.1 | 14.5 |

(RMSE. ARGO: 17 test-day profiles reaching ≥ 200 m.) TCHP, D26 and D20 errors are
**36–47 % below climatology**, against both GLORYS and independent floats. The cyclone-aware
loss gives the best D26 and TCHP errors against GLORYS with the lowest seed-to-seed spread, but
the gains over the no-physics model are small and mostly within seed variability; against ARGO
the variants are not distinguishable. **MLD is not improved over climatology** (limitation).

### Independent ARGO floats (never used in training), temperature profiles

| model | RMSE all 92 days (106 profiles) | RMSE test days only (17 profiles) |
|---|---|---|
| climatology | 1.030 | 1.245 |
| U-Net | 0.714 ± 0.023 | 0.947 ± 0.083 |
| ViT+FNO, no physics | 0.703 ± 0.026 | 0.896 ± 0.080 |
| **ViT+FNO, cyclone-aware** | 0.701 ± 0.036 | 0.917 ± 0.128 |

"All days" includes training days (the model saw GLORYS, never ARGO, on those days). The
~+0.2 °C bias vs ARGO also appears in climatology, i.e. it is inherited from GLORYS.

### Uncertainty calibration (3-seed ensemble of the main model, test days)

| against | 68 % interval | 90 % interval | 95 % interval |
|---|---|---|---|
| GLORYS | 63 % | 83 % | 89 % |
| ARGO, model σ only | 59 % | 85 % | 92 % |
| ARGO, model σ + representativeness | 80 % | 98 % | 99 % |

Intervals are slightly over-confident against GLORYS and, with the representativeness term,
conservative against ARGO: true ARGO coverage of the nominal 90 % interval lies between 85 % and
98 %.

### Case study: deep depression, NE Bay of Bengal, 1 Aug 2023

Satellite SST in the storm box drops by ~0.6 °C on 1 Aug. GLORYS shows TCHP falling from
~81 to ~70 kJ/cm² and staying low for two weeks. OceanEmbed (satellite inputs only) captures the
**initial drop** on 1–2 Aug but then **recovers too quickly** (overestimating TCHP by
~5 kJ/cm² and D26 by ~3 m), and predicts a deepening mixed layer where GLORYS shows shoaling
(likely rain-induced freshening). Daily box-mean correlation with GLORYS over 18 Jul–15 Aug:
TCHP 0.94, D26 0.88, MLD 0.50 (July days are training days). Interpretation: with 61 training
days and almost no storms, the model has not learned the post-storm ocean response; a
multi-year training set containing many cyclones is the main next step.

### Relation to prior work

Satellite-based TCHP estimation for the Indian Ocean is established: e.g. altimetry-based
estimates (RMSE ≈ 21 kJ/cm² vs in-situ) and a neural-network TCHP model from SSH anomaly, SST and
climatological D26 (Ali et al., IEEE JSTARS 2012). Those predict TCHP as a single 2-D field.
OceanEmbed instead reconstructs the full 35-level temperature field and trains it so that the
TCHP/D26/D20/MLD derived from that field are accurate, and attaches a calibrated uncertainty.
Numbers are not directly comparable across studies (different periods, data and validation sets).

## v2.1 — what is new

**1. Cyclone-aware physics losses** (`physics.py`, `--phys_mode cyclone`). The v2 constraint losses
(stability, MLD homogeneity) were already satisfied by a model fitted to GLORYS, so they could
not improve accuracy: a stable-but-wrong profile costs nothing. v2.1 adds differentiable (soft)
versions of the ocean-state quantities used for cyclone and ocean forecasting and matches them
to the target:

| quantity | meaning | soft form |
|---|---|---|
| D26 | depth of the 26 °C isotherm | Σ dz · sigmoid((T − 26)/τ) |
| TCHP | Tropical Cyclone Heat Potential (kJ/cm²) | ρ c_p Σ dz · softplus(T − 26) |
| D20 | thermocline (20 °C isotherm) depth | Σ dz · sigmoid((T − 20)/τ) |
| MLD | mixed layer depth | Σ dz · sigmoid((T − (T₁₀ − 0.5))/τ) |

Stability and steric-height/SSH consistency are kept. Evaluation uses the exact (hard)
definitions in `products.py`, against GLORYS and against the same quantities computed from ARGO.

**2. Calibrated uncertainty** (`uncertainty.py`). A heteroscedastic head predicts a per-point
temperature variance (trained with a Gaussian NLL on a detached mean, so it cannot degrade the
mean). Each seed's σ is rescaled by one factor fitted on validation days; the 3-seed ensemble
combines within-model variance and between-model spread. Coverage of 50–95 % intervals is
checked on test days against GLORYS and against ARGO. For ARGO, the float-vs-model-cell
representativeness error (ARGO − GLORYS RMS per depth band, fitted on non-test days) is added
in quadrature and reported separately. Final output: `temperature` + `temperature_std`.

**3. Ocean memory** (`data.py`, `history_days=2`): surface fields from the previous 2 days are
inputs, since the subsurface responds to forcing with a lag. Only past/present surface data.

**4. Real-event case study** (`casestudy.py`): the deep depression over the NE Bay of Bengal on
1 Aug 2023 (IMD; 21.2 °N, 91.2 °E). Box-mean daily SST, MLD, D26 and TCHP: satellite-only
reconstruction vs GLORYS vs ARGO floats, plus ΔTCHP maps. Note: 18–31 July are training days;
August days are validation/test days.

| run (3 seeds each) | encoder | physics |
|---|---|---|
| `unet` | CNN | none |
| `oceanembed_nophys` | ViT + FNO | none |
| `oceanembed_constraints` | ViT + FNO | v2 constraints |
| `oceanembed_cyclone` | ViT + FNO | cyclone-aware (main) |

## Physics-informed constraints (v2)

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
