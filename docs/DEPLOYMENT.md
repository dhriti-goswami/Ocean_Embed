# Deployment

How OceanEmbed's outputs reach users, and how to run it as a daily product.

## What gets deployed

| artifact | produced by | size / format |
|---|---|---|
| trained weights | `python -m oceanembed train` → `best.pt` (one per seed) | ~15 MB each (3.73 M float32 parameters + normalization) |
| daily 3D temperature + uncertainty | `python -m oceanembed uncertainty ... --outputs <dir>` | CF-1.8 NetCDF, `temperature`, `temperature_std` on `(time, depth, latitude, longitude)` |
| latent embeddings | same step | NetCDF, `embedding (time, channel, y, x)` |
| cyclone products | `oceanembed.physics.products.all_products()` on the NetCDF | TCHP, D26, D20, MLD per cell and day |

## Integration with the team application

The team's FastAPI backend reads a NetCDF with a `temperature` variable on
`(time, depth, latitude, longitude)` and serves depth slices and click-to-profile requests to
the React frontend. OceanEmbed's output uses exactly these names; during development the
output file was loaded with the backend's own profile and layer extraction functions to
confirm compatibility. The backend opens a **Zarr** store (`DATA_PATH`, default
`data/ocean_temperature.zarr`, relative to where the backend runs), so convert the NetCDF:

```bash
python -m oceanembed export-backend --nc <outputs>/oceanembed_temperature_daily_0p25.nc \
    --out ../backend/data/ocean_temperature.zarr
```

The exported store was loaded with `xr.open_zarr` (as `backend/app/dependencies.py` does) and
queried with the backend's `extract_profile` / `extract_layer`: both return the expected
35-level profile and 69 × 81 layer.

`temperature_std` is an extra variable the backend does not read yet; exposing it as a
confidence layer in the frontend is the natural next step.

## Precompute, don't predict on request

Inputs arrive daily and the grid is small (69 × 81 × 35), so the whole domain is predicted once
per day and stored; requests then only read the file. Prediction for all 92 days takes seconds
on a GPU and is feasible on a CPU.

## Daily operational loop (proposed, not yet automated)

```
1. download yesterday's OSTIA, SMOS/SMAP, DUACS, OSCAR, CCMP fields for the box
2. regrid to the 0.25 deg grid and append to the cube (same steps as the team data notebook)
3. run the 3-seed ensemble on the new day (needs the previous 2 days as context)
4. append temperature, temperature_std and products to the NetCDF
5. python -m oceanembed export-backend  ->  backend reads the new day
```

Notes:
- Satellite products have different latencies (near-real-time vs reprocessed versions);
  an operational run must use the near-real-time versions, whose statistics can differ from
  the reprocessed data the model was trained on. This has not been tested.
- The model has seen only June–August 2023. Running it outside that season or region is
  extrapolation; retrain on multiple years first (see [`PREREGISTRATION.md`](PREREGISTRATION.md)).

## Environments

- **Colab (tested):** `notebooks/run_pipeline_colab.ipynb`, T4 GPU.
- **Local (tested on CPU with the synthetic cube):** `make install && make smoke`.
