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
