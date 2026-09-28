import numpy as np
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

from oceanembed.export_backend import export


def test_backend_zarr_has_expected_layout(tmp_path):
    t = np.arange("2023-08-01", "2023-08-03", dtype="datetime64[D]")
    ds = xr.Dataset({"temperature": (("time", "depth", "latitude", "longitude"),
                                     np.random.rand(2, 35, 4, 5).astype("float32"))},
                    coords={"time": t, "depth": np.linspace(0.5, 902, 35),
                            "latitude": np.arange(4.0), "longitude": np.arange(5.0)})
    nc = tmp_path / "out.nc"; ds.to_netcdf(nc)
    z = xr.open_zarr(export(str(nc), str(tmp_path / "ocean_temperature.zarr")))
    assert z.temperature.dims == ("time", "depth", "latitude", "longitude")
    assert np.allclose(z.temperature.values, ds.temperature.values)


def test_rejects_wrong_dims(tmp_path):
    ds = xr.Dataset({"temperature": (("depth", "time"), np.zeros((3, 2)))})
    nc = tmp_path / "bad.nc"; ds.to_netcdf(nc)
    with pytest.raises(ValueError):
        export(str(nc), str(tmp_path / "x.zarr"))


def test_products_added_as_daily_2d_fields(tmp_path):
    from oceanembed.data.synthetic import GLORYS_DEPTHS as Z
    prof = np.interp(Z, [0, 30, 60, 120, 900], [29, 28.9, 26, 20, 8]).astype("float32")
    T = np.broadcast_to(prof[None, :, None, None], (2, len(Z), 3, 4)).copy()
    T[:, :, 0, 0] = np.nan                                   # a land cell
    t = np.arange("2023-08-01", "2023-08-03", dtype="datetime64[D]")
    ds = xr.Dataset({"temperature": (("time", "depth", "latitude", "longitude"), T)},
                    coords={"time": t, "depth": Z, "latitude": np.arange(3.0), "longitude": np.arange(4.0)})
    nc = tmp_path / "p.nc"; ds.to_netcdf(nc)
    z = xr.open_zarr(export(str(nc), str(tmp_path / "p.zarr")))
    assert {"tchp", "d26", "d20", "mld"} <= set(z.data_vars)
    assert z.tchp.dims == ("time", "latitude", "longitude")
    assert abs(float(z.d26[0, 1, 1]) - 60) < 1.5 and 50 < float(z.tchp[0, 1, 1]) < 60
    assert np.isnan(float(z.tchp[0, 0, 0]))                  # land stays empty
