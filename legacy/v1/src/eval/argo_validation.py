
"""ARGO independent validation for OceanEmbed."""
import numpy as np


def fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max,
                          date_min, date_max):
    from argopy import DataFetcher
    fetcher = DataFetcher(mode="standard")
    ds = fetcher.region(
        [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]
    ).load().data
    return ds


def match_argo_to_predictions(argo_ds, predict_fn, model_lat, model_lon, model_times,
                                target_depths):
    lat_vals = argo_ds["LATITUDE"].values
    lon_vals = argo_ds["LONGITUDE"].values
    time_vals = argo_ds["TIME"].values
    pres_vals = argo_ds["PRES"].values
    temp_vals = argo_ds["TEMP"].values

    profile_keys = list(zip(lat_vals, lon_vals, time_vals))
    unique_keys = sorted(set(profile_keys), key=lambda k: k[2])

    matched_true = []
    matched_pred = []
    skipped_out_of_bounds = 0
    skipped_no_time_match = 0
    skipped_too_few_levels = 0

    for lat0, lon0, time0 in unique_keys:
        idx = [i for i, k in enumerate(profile_keys) if k == (lat0, lon0, time0)]
        profile_pres = pres_vals[idx]
        profile_temp = temp_vals[idx]

        valid = ~np.isnan(profile_pres) & ~np.isnan(profile_temp)
        if valid.sum() < 3:
            skipped_too_few_levels += 1
            continue

        if not (model_lat.min() <= lat0 <= model_lat.max() and
                model_lon.min() <= lon0 <= model_lon.max()):
            skipped_out_of_bounds += 1
            continue

        time_diffs = np.abs(model_times - np.datetime64(time0))
        best_day_idx = np.argmin(time_diffs)
        if time_diffs[best_day_idx] > np.timedelta64(1, "D"):
            skipped_no_time_match += 1
            continue

        lat_idx = np.argmin(np.abs(model_lat - lat0))
        lon_idx = np.argmin(np.abs(model_lon - lon0))

        argo_at_target_depths = np.interp(
            target_depths, profile_pres[valid], profile_temp[valid],
            left=np.nan, right=np.nan
        )

        model_pred_profile = predict_fn(best_day_idx)[:, lat_idx, lon_idx]

        both_valid = ~np.isnan(argo_at_target_depths)
        if both_valid.sum() == 0:
            continue

        matched_true.append(argo_at_target_depths)
        matched_pred.append(model_pred_profile)

    print(f"Matched {len(matched_true)} ARGO profiles.")
    print(f"Skipped: {skipped_out_of_bounds} out of spatial bounds, "
          f"{skipped_no_time_match} no time match within 1 day, "
          f"{skipped_too_few_levels} too few valid levels")

    return np.array(matched_true), np.array(matched_pred)


def compute_argo_metrics(matched_true, matched_pred, target_depths):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from metrics import _metrics_flat

    n_depth = matched_true.shape[1]
    valid_overall = ~np.isnan(matched_true) & ~np.isnan(matched_pred)
    overall = _metrics_flat(matched_true[valid_overall], matched_pred[valid_overall])

    per_depth = []
    for d in range(n_depth):
        t = matched_true[:, d]
        p = matched_pred[:, d]
        valid = ~np.isnan(t) & ~np.isnan(p)
        if valid.sum() < 2:
            per_depth.append({"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                               "correlation": float("nan"), "bias": float("nan"), "n": int(valid.sum())})
        else:
            per_depth.append(_metrics_flat(t[valid], p[valid]))

    return {"overall": overall, "per_depth": per_depth}
