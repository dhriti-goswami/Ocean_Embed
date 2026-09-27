# Results

Written by `notebooks/run_pipeline_colab.ipynb` or `bash scripts/run_experiments.sh <cube.nc>`.
The numbers are summarized in [`docs/RESULTS.md`](../docs/RESULTS.md).

| file | content |
|---|---|
| `glorys_test_overall.csv` | per run: RMSE, MAE, bias, R², inversion rates, TCHP/D26/D20/MLD errors |
| `glorys_test_per_depth.csv` | RMSE and bias per depth level, one column per run |
| `argo_summary.csv`, `argo_all_days_*.csv`, `argo_test_days_only_*.csv` | ARGO profile metrics by depth band |
| `argo_products_summary.csv` | TCHP/D26/D20/MLD vs ARGO |
| `uncertainty_coverage.csv`, `uncertainty_by_depth.csv` | calibration of the ensemble uncertainty |
| `casestudy_timeseries.csv`, `casestudy_stats.csv` | 1 Aug 2023 deep-depression case study |
| `fig_*.png` | error by depth, training curves, ARGO profiles, depth slices, calibration, case study |

The final gridded product (`oceanembed_temperature_daily_0p25.nc`, variables `temperature`
and `temperature_std`) and checkpoints go to `artifacts/` and are not committed.
