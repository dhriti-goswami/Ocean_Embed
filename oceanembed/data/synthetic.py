"""Write a fake cube with the SAME variable names, dims and shapes as the team's
complete_model_ready_dataset.nc (92 days, 69x81 at 0.25 deg, 35 GLORYS depths),
so the whole pipeline can be tested without the real data."""
import sys
import numpy as np
import pandas as pd
import xarray as xr

GLORYS_DEPTHS = np.array([0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.93, 9.573, 11.405, 13.467,
                          15.81, 18.496, 21.599, 25.211, 29.445, 34.434, 40.344, 47.374, 55.764,
                          65.807, 77.854, 92.326, 109.729, 130.666, 155.851, 186.126, 222.475,
                          266.04, 318.127, 380.213, 453.938, 541.089, 643.567, 763.333, 902.339],
                         dtype=np.float32)


def make(path, n_days=92, with_salinity=False, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2023-06-01", periods=n_days, freq="D")
    lat = np.arange(5.0, 22.0 + 0.25, 0.25)
    lon = np.arange(80.0, 100.0 + 0.25, 0.25)
    H, W, D = len(lat), len(lon), len(GLORYS_DEPTHS)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    land = (LO < 80.8 + (LA - 5) * 0.08) | ((LA > 20.5) & (LO > 86)) | (LO > 97.5 + (22 - LA) * 0.05)
    seabed = 200 + 3000 * np.clip((LA - 5) / 17 * 0 + (LO - 81) / 6, 0, 1) * np.clip((21 - LA) / 3, 0, 1)

    t = np.arange(n_days)[:, None, None]
    eddy = np.sin(LA / 1.3 + t / 9.0) * np.cos(LO / 1.7 - t / 13.0)
    sst_c = 29.0 - 0.08 * (LA - 5) + 0.6 * eddy + 0.1 * rng.standard_normal((n_days, H, W))
    sla = 0.12 * eddy + 0.02 * rng.standard_normal((n_days, H, W))
    sss = 33.0 - 0.15 * (LA - 5) + 0.3 * eddy
    thermo = 80 + 60 * eddy                                    # thermocline depth follows SLA
    z = GLORYS_DEPTHS[None, :, None, None]
    temp = (sst_c[:, None] - 0.5 * np.clip(z - 20, 0, None) / 20
            - 15 * (1 + np.tanh((z - thermo[:, None]) / 60)) / 2
            - 4 * np.clip(z - 300, 0, None) / 600)
    temp = np.maximum(temp, 6.0).astype(np.float32)
    temp[:, (z[0, :, 0, 0][:, None, None] > seabed[None])] = np.nan
    temp[:, :, land] = np.nan

    def f2(a):
        a = a.astype(np.float32).copy(); a[:, land] = np.nan; return a

    ds = xr.Dataset(
        {
            "analysed_sst": (("time", "lat", "lon"), f2(sst_c + 273.15)),
            "sos": (("time", "lat", "lon"), f2(sss)),
            "sla": (("time", "lat", "lon"), f2(sla)),
            "adt": (("time", "lat", "lon"), f2(sla + 0.9)),
            "u": (("time", "lat", "lon"), f2(0.3 * np.gradient(eddy, axis=1))),
            "v": (("time", "lat", "lon"), f2(-0.3 * np.gradient(eddy, axis=2))),
            "uwnd": (("time", "lat", "lon"), f2(6 + rng.standard_normal((n_days, H, W)))),
            "vwnd": (("time", "lat", "lon"), f2(3 + rng.standard_normal((n_days, H, W)))),
            "ascat_uwnd": (("time", "lat", "lon"), f2(np.full((n_days, H, W), 1.8))),
            "ascat_vwnd": (("time", "lat", "lon"), f2(np.full((n_days, H, W), -2.0))),
            "thetao": (("time", "depth", "lat", "lon"), temp),
        },
        coords={"time": time, "lat": lat, "lon": lon, "depth": GLORYS_DEPTHS},
    )
    if with_salinity:
        so = (sss[:, None] + 1.5 * (1 - np.exp(-z / 80))).astype(np.float32)
        so[np.isnan(temp)] = np.nan
        ds["so"] = (("time", "depth", "lat", "lon"), so)
    ds.to_netcdf(path)
    return path


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic_cube.nc",
         with_salinity="--salinity" in sys.argv)
