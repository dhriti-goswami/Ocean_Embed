# Data

Nothing in `data/raw/` or `data/processed/` is committed (size). Put the harmonized cube at
`data/processed/cube.nc`, or pass any path with `--data`.

## Harmonized cube (model input)

Produced by the team's data notebook (downloads + xESMF regridding + daily alignment).
One NetCDF file, daily, 0.25°, Bay of Bengal (5–22 °N, 80–100 °E), 1 Jun – 31 Aug 2023.

| variable | dims | source | notes |
|---|---|---|---|
| `analysed_sst` | time, lat, lon | OSTIA | Kelvin or °C (converted automatically) |
| `sos` | time, lat, lon | SMOS/SMAP | PSU |
| `sla`, `adt` | time, lat, lon | DUACS | m |
| `u`, `v` | time, lat, lon | OSCAR | m/s |
| `uwnd`, `vwnd` | time, lat, lon | CCMP v3.1 (daily mean) | m/s |
| `ascat_uwnd`, `ascat_vwnd` | time, lat, lon | ASCAT L2 | present in the cube, **not used** |
| `thetao` | time, depth, lat, lon | GLORYS12 | °C, 35 levels 0.49–902 m (target) |
| `so` *(optional)* | time, depth, lat, lon | GLORYS12 | if present, salinity is also predicted |

`latitude`/`longitude` names are accepted as well. Shapes of the current cube:
`time 92, lat 69, lon 81, depth 35`.

In Colab the notebook downloads the team's shared copy automatically.

## ARGO

Fetched at run time from Ifremer ERDDAP (`ArgoFloats`, QC flag 1) and cached to
`results/argo_cache.csv`. Nothing to download manually.

## Synthetic cube (tests)

`python -m oceanembed synthetic data/processed/synthetic.nc` writes a fake cube with the same
variable names and shapes, used by the tests and `make smoke`.
