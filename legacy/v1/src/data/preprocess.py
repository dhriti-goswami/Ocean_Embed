
"""OceanEmbed preprocessing pipeline."""
import numpy as np
import xarray as xr

TARGET_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float32,
)


def load_and_regrid(temp_path, surface_path, wind_path, gebco_path):
    ds_temp = xr.open_dataset(temp_path)
    ds_surface = xr.open_dataset(surface_path)
    ds_wind = xr.open_dataset(wind_path)
    ds_gebco = xr.open_dataset(gebco_path)

    target_lat = ds_wind.latitude
    target_lon = ds_wind.longitude

    temp_r = ds_temp.interp(latitude=target_lat, longitude=target_lon, method="linear")
    surface_r = ds_surface.interp(latitude=target_lat, longitude=target_lon, method="linear")
    gebco_r = ds_gebco.interp(lat=target_lat.rename({"latitude": "lat"}).values,
                               lon=target_lon.rename({"longitude": "lon"}).values,
                               method="nearest")
    gebco_r = gebco_r.rename({"lat": "latitude", "lon": "longitude"})

    return temp_r, surface_r, ds_wind, gebco_r


def harmonize_time(ds_wind):
    daily = ds_wind.resample(valid_time="1D").mean()
    daily = daily.rename({"valid_time": "time"})
    return daily


def interpolate_depths(ds_temp_regridded):
    return ds_temp_regridded.interp(
        depth=TARGET_DEPTHS, method="linear", kwargs={"fill_value": "extrapolate"}
    )


def build_ocean_mask(temp_at_target_depths, X):
    y_valid = ~np.isnan(temp_at_target_depths.thetao.values)
    y_valid_all = y_valid.all(axis=(0, 1))
    x_valid = ~np.isnan(X)
    x_valid_all = x_valid.all(axis=(0, 1))
    return y_valid_all & x_valid_all


def stack_inputs(surface_regridded, wind_daily, gebco_regridded):
    n_time = surface_regridded.sizes["time"]
    sss = surface_regridded.so.isel(depth=0).values
    ssh = surface_regridded.zos.values
    wind_u = wind_daily.u10.values
    wind_v = wind_daily.v10.values
    bathy = gebco_regridded.elevation.values
    bathy_broadcast = np.broadcast_to(bathy, (n_time,) + bathy.shape)
    channels = np.stack([sss, ssh, wind_u, wind_v, bathy_broadcast], axis=1)
    channel_names = ["sss", "ssh", "wind_u", "wind_v", "bathymetry"]
    return channels, channel_names


def run_pipeline(temp_path, surface_path, wind_path, gebco_path):
    temp_r, surface_r, ds_wind, gebco_r = load_and_regrid(
        temp_path, surface_path, wind_path, gebco_path
    )
    wind_daily = harmonize_time(ds_wind)
    temp_15depth = interpolate_depths(temp_r)

    common_times = np.intersect1d(
        np.intersect1d(temp_15depth.time.values, surface_r.time.values),
        wind_daily.time.values,
    )
    temp_15depth = temp_15depth.sel(time=common_times)
    surface_r = surface_r.sel(time=common_times)
    wind_daily = wind_daily.sel(time=common_times)

    X, channel_names = stack_inputs(surface_r, wind_daily, gebco_r)
    Y = temp_15depth.thetao.values
    ocean_mask = build_ocean_mask(temp_15depth, X)

    return X, Y, ocean_mask, channel_names, common_times
