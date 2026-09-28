"""Bundle the result tables into one JSON file for the frontend's Validation page
(served by the backend, e.g. at /api/v1/validation).

    python -m oceanembed export-results --results results --out results/validation.json

Seeds are aggregated: every model is reported as mean and std over its `_s<seed>` runs.
NaN becomes null, so the file is valid JSON for browsers. Missing tables are skipped.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MAIN_MODEL = "oceanembed_cyclone"
LABELS = {
    "climatology": "Climatology (baseline)",
    "unet": "U-Net",
    "oceanembed_nophys": "ViT+FNO, no physics",
    "oceanembed_constraints": "ViT+FNO, v2 constraints",
    "oceanembed_cyclone": "OceanEmbed (main)",
    "oceanembed_pinn": "ViT+FNO + v2 constraints, no history",
}


def _clean(x):
    """Recursively turn NaN/inf and numpy types into JSON-safe values."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _model_type(name):
    return pd.Series([name]).str.replace(r"_s\d+$", "", regex=True).iloc[0]


def _agg(df, cols, by="model", extra_by=()):
    """mean/std over seeds -> list of records."""
    df = df.copy()
    df["model_type"] = df[by].map(_model_type)
    keys = list(extra_by) + ["model_type"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(keys)
    out = []
    for key, grp in g:
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        rec["label"] = LABELS.get(rec["model_type"], rec["model_type"])
        rec["n_seeds"] = int(len(grp))
        for c in cols:
            rec[c] = {"mean": grp[c].mean(), "std": grp[c].std() if len(grp) > 1 else None}
        out.append(rec)
    return out


def _read(results, name):
    p = os.path.join(results, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def build(results):
    out = {
        "region": {"lat": [5.0, 22.0], "lon": [80.0, 100.0], "name": "Bay of Bengal"},
        "period": {"start": "2023-06-01", "end": "2023-08-31",
                   "train": ["2023-06-01", "2023-07-31"], "val": ["2023-08-01", "2023-08-15"],
                   "test": ["2023-08-16", "2023-08-31"]},
        "main_model": MAIN_MODEL,
        "baseline": "climatology",
        "notes": [
            "Mean and std over 3 training seeds.",
            "Climatology = training-period mean at each cell and depth.",
            "ARGO floats were never used in training; 'all_days' includes training days.",
            "Results were not pre-registered; proof of concept.",
        ],
    }
    g = _read(results, "glorys_test_overall.csv")
    if g is not None:
        out["glorys_test"] = _agg(g, ["rmse", "mae", "bias", "r2", "deep_T_inv_pct",
                                      "TCHP_kJcm2_rmse", "D26_m_rmse", "D20_m_rmse", "MLD_m_rmse"])
    pd_ = _read(results, "glorys_test_per_depth.csv")
    if pd_ is not None:
        depth = pd_["depth_m"].tolist()
        per_model = {}
        for c in [c for c in pd_.columns if c.startswith("rmse_")]:
            per_model.setdefault(_model_type(c[5:]), []).append(pd_[c].values)
        out["glorys_rmse_by_depth"] = {
            "depth_m": depth,
            "models": {m: {"label": LABELS.get(m, m), "mean": np.mean(v, 0).tolist(),
                           "std": (np.std(v, 0, ddof=1).tolist() if len(v) > 1 else None)}
                       for m, v in per_model.items()},
        }
    a = _read(results, "argo_summary.csv")
    if a is not None:
        out["argo"] = _agg(a, ["profiles", "points", "rmse", "mae", "bias", "r2"], extra_by=("subset",))
    ap = _read(results, "argo_products_summary.csv")
    if ap is not None:
        out["argo_products"] = _agg(ap, ["profiles", "TCHP_kJcm2_rmse", "D26_m_rmse", "D20_m_rmse",
                                         "MLD_m_rmse"], extra_by=("subset",))
    u = _read(results, "uncertainty_coverage.csv")
    if u is not None:
        out["uncertainty_coverage"] = u.to_dict(orient="records")
    ub = _read(results, "uncertainty_by_depth.csv")
    if ub is not None:
        out["uncertainty_by_depth"] = ub.to_dict(orient="records")
    cs = _read(results, "casestudy_timeseries.csv")
    if cs is not None:
        stats = _read(results, "casestudy_stats.csv")
        out["case_study"] = {
            "event": {"name": "Deep depression, NE Bay of Bengal", "date": "2023-08-01",
                      "lat": 21.2, "lon": 91.2, "source": "IMD press release, 1 Aug 2023"},
            "box": {"lat": [18.5, 22.0], "lon": [87.5, 92.5]},
            "summary": ("Captures the initial heat loss on 1-2 Aug but recovers too quickly "
                        "afterwards; post-storm response not learned from 61 training days."),
            "timeseries": cs.assign(date=cs["date"].astype(str).str[:10]).to_dict(orient="list"),
            "stats": (stats.rename(columns={stats.columns[0]: "quantity"}).to_dict(orient="records")
                      if stats is not None else None),
        }
    return _clean(out)


def export(results, out_path):
    data = build(results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1, allow_nan=False)
    return out_path, data


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m oceanembed export-results")
    p.add_argument("--results", default="results", help="folder with the CSV result tables")
    p.add_argument("--out", default="results/validation.json")
    a = p.parse_args(argv)
    path, data = export(a.results, a.out)
    print("wrote", path, "| sections:", [k for k in data if k not in ("region", "period", "notes")])


if __name__ == "__main__":
    main()
