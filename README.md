# OceanEmbed — satellite-embedding reconstruction of subsurface ocean temperature

OceanEmbed looks only at **surface satellite observations** and reconstructs the ocean's
**temperature from the surface down to ~900 m**, every day, on a 0.25° grid over the
**Bay of Bengal**, with a **calibrated uncertainty** at every point and the
**cyclone-relevant ocean products** derived from it (cyclone heat potential, 26 °C and 20 °C
isotherm depths, mixed layer depth).

Built for **Smart India Hackathon 2026**, Problem Statement **26066** (INCOIS), team
**BitShifters98**.

> Current system: **v2.1** (`oceanembed/`). The earlier CNN baseline (v1, `src/`) is kept for
> reference at the end of this page.

---

## Headline results (Bay of Bengal, Jun–Aug 2023, held-out test days 16–31 Aug)

| | OceanEmbed | climatology baseline | improvement |
|---|---|---|---|
| Temperature RMSE, 0–902 m (vs GLORYS) | **0.63 °C** | 0.97 °C | **−36 %** |
| Cyclone heat potential (TCHP) RMSE vs **independent ARGO floats** | **16.5 kJ/cm²** | 29.8 kJ/cm² | **−45 %** |
| 26 °C isotherm depth RMSE vs GLORYS | **8.4 m** | 15.0 m | −44 % |
| Temperature RMSE vs ARGO floats (test days) | **0.92 °C** | 1.25 °C | −26 % |
| Share of observations inside the 90 % uncertainty interval | **83–87 %** | – | – |

All numbers are means over 3 training seeds. Full tables, ablations and caveats:
[`oceanembed/README.md`](oceanembed/README.md#results-bay-of-bengal-1-jun--31-aug-2023).

---

## How it works

```
 SATELLITE INPUTS (today + previous 2 days)        TRAINING TARGET
  SST  - OSTIA            SSS - SMOS/SMAP            GLORYS reanalysis temperature
  SSH  - DUACS (SLA, ADT) currents - OSCAR           35 levels, 0.5-902 m
  winds - CCMP            + date, lat/lon
            |
            v
 PREPROCESSING  regrid to 0.25 deg, daily alignment, land/seabed masks,
                contiguous time split, train-only normalization
            |
            v
 ENCODER (satellite embedding)
   Fourier Neural Operator  -> basin-scale patterns
   Vision Transformer       -> long-range links between regions
   CNN stem                 -> local detail
   => latent ocean embedding
            |
            v
 DECODER (U-Net)  -> temperature at 35 depths  + its uncertainty
            |
 PHYSICS-INFORMED TRAINING
   cyclone-aware losses: TCHP, D26, D20, MLD (differentiable versions)
   static stability, steric-height / sea-level consistency
            |
            v
 OUTPUTS   daily 0.25 deg NetCDF: temperature + temperature_std
           cyclone products: TCHP, D26, D20, MLD
           latent embeddings
            |
 VALIDATION  held-out days vs GLORYS, independent ARGO floats (profiles and products),
             uncertainty calibration, real-event case study (deep depression, 1 Aug 2023)
```

### Inputs (all from the problem statement's product list)

| variable | product | native resolution |
|---|---|---|
| Sea surface temperature | OSTIA | 0.05°, daily |
| Sea surface salinity | SMOS / SMAP | 0.125°, daily |
| Sea level anomaly, absolute dynamic topography | DUACS | 0.125–0.25°, daily |
| Surface currents | OSCAR | 0.25°, daily |
| 10 m winds | CCMP v3.1 | 0.25°, 6-hourly → daily |
| Training target | GLORYS12 reanalysis `thetao` | 1/12° → 0.25°, 35 levels |
| Independent validation | ARGO floats (Ifremer ERDDAP, QC = good) | profiles |

The harmonized cube (daily, 0.25°, 69 × 81 grid) is produced by the team's data notebook.
ASCAT L2 swaths are not used as inputs (single daily swaths are not gridded fields; CCMP
already assimilates ASCAT).

### What is distinctive

- **Full 3D field + consistent cyclone products.** Earlier satellite TCHP methods for the
  Indian Ocean estimate TCHP directly as a 2D field. OceanEmbed reconstructs the whole
  temperature column and is trained so that the TCHP, D26, D20 and MLD *derived from that
  column* are accurate.
- **Uncertainty checked against real floats.** A heteroscedastic head plus a 3-seed ensemble,
  calibrated on validation days and tested against held-out GLORYS and ARGO, including the
  float-vs-model-cell representativeness error.
- **Honest evaluation.** Contiguous time split, climatology baseline, 3 seeds per model,
  ablations (U-Net vs ViT+FNO, with/without physics, with/without previous-day inputs),
  per-depth and per-product metrics.

### What the ablations show

- ViT + FNO encoder beats a plain U-Net slightly and is much more stable across seeds.
- Previous-day surface inputs reduce error by ~5 %.
- Physics losses keep the output physically stable but do **not** significantly improve
  overall accuracy; the cyclone-aware loss gives the best D26/TCHP errors, within seed
  variability.

### Limitations

- 3 months (summer monsoon) of training data, one basin.
- Mixed layer depth is not better than climatology.
- In the 1 Aug 2023 deep-depression case study the model captures the initial heat loss but
  recovers too quickly afterwards: it has not learned the post-storm response from 61
  training days. More years (with many cyclones) is the main next step.
- GLORYS assimilates ARGO, so ARGO is independent of the model's inputs and training, but not
  perfectly independent of its training target.

---

## Quick start

**Colab (recommended, GPU, ~20–25 min):** open
[`notebooks/run_pipeline_colab.ipynb`](https://colab.research.google.com/github/dhriti-goswami/Ocean_Embed/blob/pinn-pipeline/notebooks/run_pipeline_colab.ipynb),
choose a T4 GPU runtime, **Run all**. It downloads the data cube, trains 4 models × 3 seeds,
evaluates, validates against ARGO, calibrates uncertainty, runs the case study, draws all
figures and packs everything into one zip.

**Command line:**

```bash
git clone -b pinn-pipeline https://github.com/dhriti-goswami/Ocean_Embed.git
cd Ocean_Embed && pip install -r requirements.txt

# train the main model (repeat with --seed 1, 2 for the ensemble)
python -m oceanembed.train --data cube.nc --out runs/oceanembed_cyclone_s0 \
    --variant oceanembed --physics --phys_mode cyclone --uncertainty --seed 0

python -m oceanembed.evaluate    --data cube.nc --runs runs/* --out results        # vs GLORYS + products
python -m oceanembed.argo        --data cube.nc --runs runs/* --out results        # vs ARGO floats
python -m oceanembed.uncertainty --data cube.nc --runs runs/oceanembed_cyclone_s* \
    --out results --outputs outputs                                                # calibrated NetCDF
python -m oceanembed.casestudy   --data cube.nc --runs runs/oceanembed_cyclone_s* --out results
```

**Smoke test without real data** (synthetic cube with the real shapes, CPU, ~2 min):
`python tests/smoke_test.py`

### Output format

`outputs/oceanembed_temperature_daily_0p25.nc` — CF-1.8 NetCDF, variables `temperature` and
`temperature_std` on `(time, depth, latitude, longitude)`, °C. This is the format the team's
FastAPI backend reads for depth slices and click-to-profile.

---

## Project structure

```
Ocean_Embed/
├── oceanembed/                    # v2.1 pipeline (current)
│   ├── data.py                    # load cube, masks, previous-day inputs, split, normalization
│   ├── models.py                  # FNO + ViT encoder, U-Net decoder, uncertainty head
│   ├── physics.py                 # cyclone-aware + stability + steric losses
│   ├── products.py                # TCHP, D26, D20, MLD (exact definitions)
│   ├── train.py                   # training loop, warm-up/ramp, checkpointing
│   ├── evaluate.py                # held-out GLORYS metrics, per depth + products
│   ├── argo.py                    # ARGO fetch (QC), matching, profile + product metrics
│   ├── uncertainty.py             # calibration, seed ensemble, coverage, final NetCDF
│   ├── casestudy.py               # 1 Aug 2023 deep-depression case study
│   ├── infer.py                   # prediction + NetCDF/embedding export
│   ├── plots.py                   # figures
│   └── README.md                  # full method, results and caveats
├── notebooks/run_pipeline_colab.ipynb
├── tests/                         # synthetic cube + end-to-end smoke test
├── src/                           # v1 CNN baseline (legacy)
├── docs/                          # v1 documentation
├── results/, checkpoints/         # v1 results and weights
└── requirements.txt
```

---

## Earlier baseline (v1, `src/`)

The first version was a plain CNN predicting 15 depth levels from GLORYS surface fields,
ERA5 winds and GEBCO bathymetry (not satellite products, and without SST), trained on 4 months
of data. It reached 0.53 °C RMSE on held-out GLORYS days and 0.96 °C against ARGO (0.76 °C after
a held-out-tested bias correction). v2 replaced its inputs with the problem statement's
satellite products and rebuilt the model; the v1 code and its documentation
([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/TRAINING_METHODOLOGY.md`](docs/TRAINING_METHODOLOGY.md),
[`docs/VALIDATION.md`](docs/VALIDATION.md)) are kept for reference. The two versions' numbers
are not directly comparable (different inputs, depth levels, periods and splits).

## Requirements

Python 3.10+, PyTorch 2.0+, xarray, netCDF4, pandas, scipy, matplotlib (see
`requirements.txt`). ARGO data is fetched directly over HTTP from Ifremer ERDDAP.

## Disclaimer

Research / hackathon prototype. Not for operational or safety-critical use.
