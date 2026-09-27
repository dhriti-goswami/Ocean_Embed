"""Convert the OceanEmbed NetCDF product into the Zarr store the team's FastAPI backend reads,
adding the cyclone / ocean-state products (TCHP, D26, D20, MLD) as 2D daily fields.

The backend (backend/app/dependencies.py) opens `settings.DATA_PATH`
(default `data/ocean_temperature.zarr`, relative to where the backend runs) with
`xr.open_zarr` and reads the `temperature` variable on (time, depth, latitude, longitude).

    python -m oceanembed export-backend --nc artifacts/outputs/oceanembed_temperature_daily_0p25.nc \
        --out ../backend/data/ocean_temperature.zarr
"""
import argparse
import os
import shutil

import numpy as np
import xarray as xr

from oceanembed.physics.products import all_products

PRODUCTS = {  # zarr name: (products key, units, long name)
    "tchp": ("TCHP_kJcm2", "kJ cm-2", "tropical cyclone heat potential (heat above the 26 degC isotherm)"),
    "d26": ("D26_m", "m", "depth of the 26 degC isotherm"),
    "d20": ("D20_m", "m", "depth of the 20 degC isotherm (thermocline)"),
    "mld": ("MLD_m", "m", "mixed layer depth (0.5 degC below the 10 m temperature)"),
}


def add_products(ds):
    """Compute TCHP, D26, D20, MLD from `temperature` and attach them as (time, lat, lon) fields."""
    T = ds["temperature"].values
    P = all_products(T, ds["depth"].values, axis=1)
    dims = ("time", "latitude", "longitude")
    for name, (key, units, long_name) in PRODUCTS.items():
        ds[name] = (dims, P[key].astype(np.float32))
        ds[name].attrs.update(units=units, long_name=long_name)
    return ds


def export(nc_path, out_path, overwrite=True, products=True):
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
    if products:
        ds = add_products(ds)
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
    p.add_argument("--no-products", action="store_true", help="skip TCHP/D26/D20/MLD")
    a = p.parse_args(argv)
    out = export(a.nc, a.out, products=not a.no_products)
    z = xr.open_zarr(out)
    print("wrote", out, "| variables:", list(z.data_vars), "| sizes:", dict(z.sizes))


if __name__ == "__main__":
    main()
