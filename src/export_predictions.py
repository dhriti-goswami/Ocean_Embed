
"""OceanEmbed prediction export -- generates the 3D cube for frontend use."""
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import OceanEmbedPredictor, TARGET_DEPTHS


def export_single_day_json(predictor, X_day, lat_array, lon_array, output_path):
    full_pred = predictor.predict(X_day[0], X_day[1], X_day[2], X_day[3], X_day[4])
    payload = {
        "depths_m": TARGET_DEPTHS,
        "lat": lat_array.tolist(),
        "lon": lon_array.tolist(),
        "temperature": np.round(full_pred, 3).tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    return full_pred


def export_multi_day_npz(predictor, X_all_days, lat_array, lon_array, dates, output_path):
    n_days = X_all_days.shape[0]
    n_depth = len(TARGET_DEPTHS)
    lat_n, lon_n = len(lat_array), len(lon_array)
    all_temps = np.zeros((n_days, n_depth, lat_n, lon_n), dtype=np.float32)
    for day_idx in range(n_days):
        X_day = X_all_days[day_idx]
        all_temps[day_idx] = predictor.predict(
            X_day[0], X_day[1], X_day[2], X_day[3], X_day[4]
        )
    np.savez_compressed(
        output_path, temperature=all_temps, depths_m=np.array(TARGET_DEPTHS),
        lat=lat_array, lon=lon_array, dates=np.array([str(d) for d in dates]),
    )
    return all_temps
