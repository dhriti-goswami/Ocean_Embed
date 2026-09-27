"""Convert the OceanEmbed NetCDF product into the Zarr store the team's FastAPI backend reads.

The backend (backend/app/dependencies.py) opens `settings.DATA_PATH`
(default `data/ocean_temperature.zarr`, relative to where the backend runs) with
`xr.open_zarr` and reads the `temperature` variable on (time, depth, latitude, longitude).

    python -m oceanembed export-backend --nc artifacts/outputs/oceanembed_temperature_daily_0p25.nc \
        --out ../backend/data/ocean_temperature.zarr
"""
import argparse
import os
import shutil

import xarray as xr


def export(nc_path, out_path, overwrite=True):
    ds = xr.open_dataset(nc_path)
    missing = {"temperature"} - set(ds.data_vars)
    if missing:
        raise KeyError(f"{nc_path} has no 'temperature' variable")
    dims = ds.temperature.dims
    if dims != ("time", "depth", "latitude", "longitude"):
        raise ValueError(f"unexpected dims {dims}; backend expects (time, depth, latitude, longitude)")
    if overwrite and os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ds = ds.load()
    enc = {}
    for v in ds.data_vars:                      # one chunk per day (no dask needed)
        ds[v].encoding.clear()
        enc[v] = {"chunks": (1,) + tuple(ds[v].shape[1:])}
    ds.to_zarr(out_path, mode="w", encoding=enc)
    ds.close()
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m oceanembed export-backend")
    p.add_argument("--nc", required=True, help="oceanembed_temperature_daily_0p25.nc")
    p.add_argument("--out", required=True, help="backend DATA_PATH, e.g. ../backend/data/ocean_temperature.zarr")
    a = p.parse_args(argv)
    print("wrote", export(a.nc, a.out))


if __name__ == "__main__":
    main()
