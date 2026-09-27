"""Independent validation against real ARGO float profiles (Ifremer ERDDAP).

ARGO is never used in training. Each profile is matched to the model grid cell and the
same calendar day, and the model's 35-level profile is interpolated to the float's own
measurement depths (pressure in dbar ~ depth in m over 0-900 m).

    python -m oceanembed.argo --data cube.nc --runs runs/unet runs/oceanembed_pinn --out results/
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
import requests

from .data import DataConfig, OceanData
from .evaluate import metrics, climatology
from .infer import load_model, predict_all

BANDS = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 500), (500, 950)]
ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"


def fetch_argo(lon_min, lon_max, lat_min, lat_max, date_min, date_max, cache=None, max_pres=1000):
    if cache and os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["time"])
        print(f"ARGO: loaded {len(df)} rows from cache {cache}")
        return df
    base_q = (f"&latitude>={lat_min}&latitude<={lat_max}&longitude>={lon_min}&longitude<={lon_max}"
              f"&time>={date_min}T00:00:00Z&time<={date_max}T23:59:59Z&pres<={max_pres}")
    attempts = [
        ("platform_number,cycle_number,latitude,longitude,time,pres,temp,pres_qc,temp_qc",
         '&temp_qc="1"&pres_qc="1"'),
        ("platform_number,cycle_number,latitude,longitude,time,pres,temp", ""),
    ]
    last_err = None
    for vars_, qc in attempts:
        try:
            r = requests.get(f"{ERDDAP}?{vars_}{base_q}{qc}", timeout=300)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
            print(f"ARGO: fetched {len(df)} rows ({'QC=1 only' if qc else 'no QC filter'})")
            break
        except Exception as e:                     # QC columns unsupported -> retry without
            last_err = e
    else:
        raise RuntimeError(f"ARGO fetch failed: {last_err}")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.dropna(subset=["pres", "temp"])
    df = df[(df.temp > -2) & (df.temp < 40)]
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def match(df, data, pred_by_day):
    """Returns long table: one row per ARGO measurement with model value at the same point.

    pred_by_day: dict {day_index: (D, H, W) array in deg C}
    """
    day_of = {np.datetime64(t, "D"): i for i, t in enumerate(data.times)}
    rows = []
    keys = ["platform_number", "cycle_number"] if "cycle_number" in df else ["platform_number", "time"]
    for _, g in df.groupby(keys):
        g = g.sort_values("pres")
        if len(g) < 5:
            continue
        di = day_of.get(np.datetime64(g.time.iloc[0], "D"))
        if di is None or di not in pred_by_day:
            continue
        lat0, lon0 = g.latitude.iloc[0], g.longitude.iloc[0]
        yi, xi = int(np.abs(data.lat - lat0).argmin()), int(np.abs(data.lon - lon0).argmin())
        if abs(data.lat[yi] - lat0) > 0.2 or abs(data.lon[xi] - lon0) > 0.2 or not data.mask2d[yi, xi]:
            continue
        prof = pred_by_day[di][:, yi, xi]
        ok = ~np.isnan(prof)
        if ok.sum() < 3:
            continue
        z = g.pres.values
        inside = z <= data.depths[ok].max()
        model_at_z = np.interp(z[inside], data.depths[ok], prof[ok])
        for zz, tt, mm in zip(z[inside], g.temp.values[inside], model_at_z):
            rows.append((g.platform_number.iloc[0], di, yi, xi, zz, tt, mm))
    return pd.DataFrame(rows, columns=["platform", "day", "yi", "xi", "depth", "argo", "model"])


def band_table(m):
    out = [dict(band=f"{a}-{b} m", **metrics(m.argo.values[(m.depth >= a) & (m.depth < b)],
                                           m.model.values[(m.depth >= a) & (m.depth < b)]))
           for a, b in BANDS]
    out.append(dict(band="ALL", **metrics(m.argo.values, m.model.values)))
    return pd.DataFrame(out)


def run(data_path, run_dirs, out_dir, cache="data/argo_cache.csv"):
    os.makedirs(out_dir, exist_ok=True)
    data = OceanData(DataConfig(path=data_path))
    t0, t1 = str(data.times[0])[:10], str(data.times[-1])[:10]
    df = fetch_argo(float(data.lon.min()), float(data.lon.max()), float(data.lat.min()),
                    float(data.lat.max()), t0, t1, cache=cache)

    all_days = np.arange(len(data.times))
    models = {"climatology": {d: climatology(data) for d in all_days}}
    for rd in run_dirs:
        model, _ = load_model(os.path.join(rd, "best.pt"))
        T, _, _ = predict_all(model, data)
        models[os.path.basename(os.path.normpath(rd))] = {d: T[d] for d in all_days}

    test_days = set(data.idx["test"].tolist())
    summary = []
    for name, pbd in models.items():
        m = match(df, data, pbd)
        m.to_csv(os.path.join(out_dir, f"argo_matches_{name}.csv"), index=False)
        for subset, mm in [("all_days", m), ("test_days_only", m[m.day.isin(test_days)])]:
            if len(mm) == 0:
                continue
            bt = band_table(mm)
            bt.to_csv(os.path.join(out_dir, f"argo_{subset}_{name}.csv"), index=False)
            a = bt[bt.band == "ALL"].iloc[0]
            summary.append(dict(model=name, subset=subset, profiles=mm.groupby(["platform", "day"]).ngroups,
                                points=int(a.n), rmse=a.rmse, mae=a.mae, bias=a.bias, r2=a.r2))
            if subset == "test_days_only" or name != "climatology":
                print(f"\n--- ARGO | {name} | {subset} ---")
                print(bt.round(3).to_string(index=False))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out_dir, "argo_summary.csv"), index=False)
    print("\n=== ARGO summary (independent observations) ===")
    print(summary.round(3).to_string(index=False))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="data/argo_cache.csv")
    a = p.parse_args(argv)
    run(a.data, a.runs, a.out, a.cache)


if __name__ == "__main__":
    main()
