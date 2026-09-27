
"""ARGO validation via direct Ifremer ERDDAP requests (no argopy needed)."""
import pandas as pd
import numpy as np
import requests
import io


def fetch_argo_csv(lon_min, lon_max, lat_min, lat_max, date_min, date_max,
                     max_pres=1100, timeout=120):
    base = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
    vars_ = "platform_number,latitude,longitude,time,pres,temp"
    query = (
        f"{base}?{vars_}"
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&time>={date_min}T00:00:00Z&time<={date_max}T23:59:59Z"
        f"&pres<={max_pres}"
        f"&distinct()"
    )
    print("Requesting:", query)
    resp = requests.get(query, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.dropna(subset=["pres", "temp"])
    return df


def match_argo_to_predictions(argo_df, predict_fn, model_lat, model_lon, model_times,
                                target_depths, time_tolerance_days=1):
    matched_true = []
    matched_pred = []
    skipped_out_of_bounds = 0
    skipped_no_time_match = 0
    skipped_too_few_levels = 0

    argo_df = argo_df.copy()
    if isinstance(argo_df["time"].dtype, pd.DatetimeTZDtype):
        argo_df["time"] = argo_df["time"].dt.tz_localize(None)

    grouped = argo_df.groupby(["platform_number", "latitude", "longitude", "time"])

    for (platform, lat0, lon0, time0), group in grouped:
        profile_pres = group["pres"].values
        profile_temp = group["temp"].values

        if len(profile_pres) < 3:
            skipped_too_few_levels += 1
            continue

        if not (model_lat.min() <= lat0 <= model_lat.max() and
                model_lon.min() <= lon0 <= model_lon.max()):
            skipped_out_of_bounds += 1
            continue

        time_diffs = np.abs(model_times - np.datetime64(time0))
        best_day_idx = np.argmin(time_diffs)
        if time_diffs[best_day_idx] > np.timedelta64(time_tolerance_days, "D"):
            skipped_no_time_match += 1
            continue

        lat_idx = np.argmin(np.abs(model_lat - lat0))
        lon_idx = np.argmin(np.abs(model_lon - lon0))

        argo_at_target_depths = np.interp(
            target_depths, profile_pres, profile_temp,
            left=np.nan, right=np.nan
        )

        model_pred_profile = predict_fn(best_day_idx)[:, lat_idx, lon_idx]

        if np.isnan(argo_at_target_depths).all():
            continue

        matched_true.append(argo_at_target_depths)
        matched_pred.append(model_pred_profile)

    print(f"Matched {len(matched_true)} ARGO profiles.")
    print(f"Skipped: {skipped_out_of_bounds} out of spatial bounds, "
          f"{skipped_no_time_match} no time match within {time_tolerance_days} day(s), "
          f"{skipped_too_few_levels} too few valid levels")

    return np.array(matched_true), np.array(matched_pred)
