# v1 baseline (legacy)

The first OceanEmbed prototype: a plain CNN predicting 15 depth levels from GLORYS surface
fields, ERA5 winds and GEBCO bathymetry (not the PS satellite products, and without SST),
4 months of data. Results: 0.53 °C RMSE on held-out GLORYS days, 0.96 °C vs ARGO (0.76 °C after
a held-out-tested bias correction). Documentation: [`docs/legacy_v1/`](../../docs/legacy_v1/).

Kept for reference only; not maintained. The code uses `from src...` imports, so run it from
this directory (`cd legacy/v1`). Numbers are not comparable with v2.1 (different inputs,
depth levels, period and split).
