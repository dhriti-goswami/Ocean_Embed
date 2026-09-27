"""End-to-end: train (smoke config) -> evaluate -> uncertainty -> NetCDF output."""
import pytest
import xarray as xr

from oceanembed.cli import main


@pytest.mark.slow
def test_end_to_end(cube, tmp_path):
    runs = [str(tmp_path / f"main_s{s}") for s in (0, 1)]
    for s, r in enumerate(runs):
        assert main(["train", "--config", "configs/smoke.yaml", "--data", cube, "--out", r, "--seed", str(s)]) == 0
    assert main(["evaluate", "--data", cube, "--runs", *runs, "--out", str(tmp_path / "res")]) == 0
    out = tmp_path / "outputs"
    import pandas as pd
    cache = tmp_path / "argo.csv"
    ds = xr.open_dataset(cube)
    # minimal fake ARGO cache so the uncertainty step runs offline
    rows = []
    for k in range(40):
        for z in (5.0, 50.0, 150.0, 400.0):
            rows.append(dict(platform_number=k, cycle_number=k, latitude=15.0, longitude=88.0,
                             time=str(ds.time.values[k * 2])[:19], pres=z,
                             temp=float(ds.thetao.sel(lat=15.0, lon=88.0, depth=z, method="nearest")
                                        .isel(time=k * 2)) + 0.2))
    pd.DataFrame(rows).to_csv(cache, index=False)
    assert main(["uncertainty", "--data", cube, "--runs", *runs, "--out", str(tmp_path / "res"),
                 "--outputs", str(out), "--argo_cache", str(cache)]) == 0
    o = xr.open_dataset(out / "oceanembed_temperature_daily_0p25.nc")
    assert {"temperature", "temperature_std"} <= set(o.data_vars)
    assert o.temperature.dims == ("time", "depth", "latitude", "longitude")
