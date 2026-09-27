# Results — Bay of Bengal, 1 Jun – 31 Aug 2023

Train 1 Jun–31 Jul, validation 1–15 Aug, **test 16–31 Aug** (never used for training or model
selection). Every model trained with 3 seeds; values are mean ± std across seeds.
Baseline "climatology" = training-period mean at each cell and depth.

## Temperature, 0–902 m, held-out test days vs GLORYS

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

## Cyclone-relevant products, test days

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

## Independent ARGO floats (never used in training), temperature profiles

| model | RMSE all 92 days (106 profiles) | RMSE test days only (17 profiles) |
|---|---|---|
| climatology | 1.030 | 1.245 |
| U-Net | 0.714 ± 0.023 | 0.947 ± 0.083 |
| ViT+FNO, no physics | 0.703 ± 0.026 | 0.896 ± 0.080 |
| **ViT+FNO, cyclone-aware** | 0.701 ± 0.036 | 0.917 ± 0.128 |

"All days" includes training days (the model saw GLORYS, never ARGO, on those days). The
~+0.2 °C bias vs ARGO also appears in climatology, i.e. it is inherited from GLORYS.

## Uncertainty calibration (3-seed ensemble of the main model, test days)

| against | 68 % interval | 90 % interval | 95 % interval |
|---|---|---|---|
| GLORYS | 63 % | 83 % | 89 % |
| ARGO, model σ only | 59 % | 85 % | 92 % |
| ARGO, model σ + representativeness | 80 % | 98 % | 99 % |

Intervals are slightly over-confident against GLORYS and, with the representativeness term,
conservative against ARGO: true ARGO coverage of the nominal 90 % interval lies between 85 % and
98 %.

## Case study: deep depression, NE Bay of Bengal, 1 Aug 2023

Satellite SST in the storm box drops by ~0.6 °C on 1 Aug. GLORYS shows TCHP falling from
~81 to ~70 kJ/cm² and staying low for two weeks. OceanEmbed (satellite inputs only) captures the
**initial drop** on 1–2 Aug but then **recovers too quickly** (overestimating TCHP by
~5 kJ/cm² and D26 by ~3 m), and predicts a deepening mixed layer where GLORYS shows shoaling
(likely rain-induced freshening). Daily box-mean correlation with GLORYS over 18 Jul–15 Aug:
TCHP 0.94, D26 0.88, MLD 0.50 (July days are training days). Interpretation: with 61 training
days and almost no storms, the model has not learned the post-storm ocean response; a
multi-year training set containing many cyclones is the main next step.

## Relation to prior work

Satellite-based TCHP estimation for the Indian Ocean is established: e.g. altimetry-based
estimates (RMSE ≈ 21 kJ/cm² vs in-situ) and a neural-network TCHP model from SSH anomaly, SST and
climatological D26 (Ali et al., 2012; see CITATIONS.md). Those predict TCHP as a single 2-D field.
OceanEmbed instead reconstructs the full 35-level temperature field and trains it so that the
TCHP/D26/D20/MLD derived from that field are accurate, and attaches a calibrated uncertainty.
Numbers are not directly comparable across studies (different periods, data and validation sets).

## Reproducing

Every number above comes from `notebooks/run_pipeline_colab.ipynb` (or
`bash scripts/run_experiments.sh <cube.nc>`), which writes the CSV tables and figures listed in
[`results/README.md`](../results/README.md). The configs used are in [`configs/`](../configs/).
