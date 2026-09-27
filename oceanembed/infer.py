"""Run a trained model over every day and write the standardized outputs.

    python -m oceanembed.infer --data cube.nc --ckpt runs/oceanembed_pinn/best.pt --out outputs/

Writes
  oceanembed_temperature_daily_0p25.nc : temperature (time, depth, latitude, longitude), deg C
                                          (+ salinity if the model predicts it)
  oceanembed_embeddings.nc             : latent ocean embedding z per day
The temperature file uses the variable/dim names the FastAPI backend reads.
"""
import argparse
import os

import numpy as np
import torch
import xarray as xr

from .data import DataConfig, OceanData
from .models import build_model


def load_model(ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["variant"], ck["in_ch"], len(ck["depths"]), tuple(ck["grid"]), ck["out_vars"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_all(model, data, indices=None, batch=8):
    """Returns T (N,D,H,W) deg C, S or None, Z latent (N,latent,h,w); land/seabed = NaN."""
    device = next(model.parameters()).device
    idx = np.arange(len(data.times)) if indices is None else np.asarray(indices)
    D = len(data.depths)
    Ts, Ss, Zs = [], [], []
    for i in range(0, len(idx), batch):
        x = torch.from_numpy(data.pad(data.X[idx[i:i + batch]])).to(device)
        z, skips = model.encode(x)
        out = data.unpad(model.decode(z, skips)).cpu().numpy()
        Ts.append(out[:, :D] * data.t_std[None, :, None, None] + data.t_mean[None, :, None, None])
        if out.shape[1] > D:
            Ss.append(out[:, D:] * data.s_std[None, :, None, None] + data.s_mean[None, :, None, None])
        Zs.append(z.cpu().numpy())
    T = np.concatenate(Ts).astype(np.float32)
    T[:, ~data.mask3d] = np.nan
    S = None
    if Ss:
        S = np.concatenate(Ss).astype(np.float32)
        S[:, ~data.mask3d] = np.nan
    return T, S, np.concatenate(Zs).astype(np.float32)


def write_outputs(data, T, S, Z, out_dir, meta=""):
    os.makedirs(out_dir, exist_ok=True)
    coords = {"time": data.times, "depth": data.depths, "latitude": data.lat, "longitude": data.lon}
    ds = xr.Dataset({"temperature": (("time", "depth", "latitude", "longitude"), T)}, coords=coords)
    ds.temperature.attrs.update(units="degC", long_name="reconstructed sea water potential temperature")
    if S is not None:
        ds["salinity"] = (("time", "depth", "latitude", "longitude"), S)
        ds.salinity.attrs.update(units="1e-3", long_name="reconstructed sea water salinity")
    ds.depth.attrs.update(units="m", positive="down")
    ds.latitude.attrs.update(units="degrees_north"); ds.longitude.attrs.update(units="degrees_east")
    ds.attrs.update(title="OceanEmbed subsurface temperature reconstruction",
                    resolution="daily, 0.25 degree", region="Bay of Bengal",
                    source="OceanEmbed (ViT+FNO encoder, U-Net decoder, physics-informed losses) "
                           "from OSTIA SST, SMOS/SMAP SSS, DUACS SSH, OSCAR currents, CCMP winds",
                    model=meta, Conventions="CF-1.8")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    p1 = os.path.join(out_dir, "oceanembed_temperature_daily_0p25.nc")
    ds.to_netcdf(p1, encoding=enc)

    ez = xr.Dataset({"embedding": (("time", "channel", "y", "x"), Z)}, coords={"time": data.times})
    ez.attrs["description"] = ("Latent ocean embedding from the satellite encoder "
                               "(each cell covers an 8x8 block of the 0.25 deg grid).")
    p2 = os.path.join(out_dir, "oceanembed_embeddings.nc")
    ez.to_netcdf(p2, encoding={"embedding": {"zlib": True}})
    return p1, p2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    data = OceanData(DataConfig(path=a.data))
    model, ck = load_model(a.ckpt)
    T, S, Z = predict_all(model, data)
    paths = write_outputs(data, T, S, Z, a.out, meta=f"{ck['variant']} physics={ck['physics']}")
    print("wrote", *paths)


if __name__ == "__main__":
    main()
