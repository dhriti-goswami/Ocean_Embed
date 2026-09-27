"""Compare trained runs on the held-out TEST days (GLORYS target): temperature per depth,
cyclone products (TCHP, D26, D20, MLD) and physics diagnostics.

    python -m oceanembed evaluate --data cube.nc --runs runs/* --out results/
"""
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
import os

import numpy as np
import pandas as pd

from oceanembed.data.cube import DataConfig, OceanData
from oceanembed.inference import load_model, predict_all, data_for
from oceanembed.physics.products import product_rmse
from oceanembed.physics.losses import density
import torch


def metrics(true, pred):
    m = ~np.isnan(true) & ~np.isnan(pred)
    t, p = true[m], pred[m]
    if t.size < 2:
        return dict(rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan, corr=np.nan, n=int(t.size))
    err = p - t
    ss_res, ss_tot = (err ** 2).sum(), ((t - t.mean()) ** 2).sum()
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                bias=float(err.mean()), r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                corr=float(np.corrcoef(t, p)[0, 1]), n=int(t.size))


def per_depth(true, pred, depths):
    """true/pred: (N, D, H, W)."""
    rows = [dict(depth_m=float(d), **metrics(true[:, k], pred[:, k])) for k, d in enumerate(depths)]
    return pd.DataFrame(rows)


def climatology(data):
    """Baseline: training-period mean at every cell and depth (no satellite information)."""
    clim = np.nanmean(data.T[data.idx["train"]], 0)
    return clim


def instability_fraction(T, S, mask3d):
    """Share of adjacent level pairs where density decreases with depth (unphysical)."""
    Tt = torch.from_numpy(np.nan_to_num(T, nan=0.0))
    St = torch.from_numpy(np.nan_to_num(S, nan=35.0)) if S is not None else None
    rho = density(Tt, St).numpy()
    pair = (mask3d[:-1] & mask3d[1:])[None].repeat(T.shape[0], 0)
    bad = (rho[:, :-1] - rho[:, 1:] > 0.01) & pair
    return float(bad.sum() / max(pair.sum(), 1))


def deep_inversion_pct(T, depths, mask3d, below=150.0, tol=0.05):
    """% of adjacent level pairs below `below` m where T increases with depth by > tol deg C."""
    pair = (mask3d[:-1] & mask3d[1:]) & (depths[1:] >= below)[:, None, None]
    bad = ((T[:, 1:] - T[:, :-1]) > tol) & pair[None]
    return float(100 * bad.sum() / max(pair.sum() * T.shape[0], 1))


def evaluate(data_path, run_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = OceanData(DataConfig(path=data_path))
    te = data.idx["test"]
    true = data.T[te].copy()
    true[:, ~data.mask3d] = np.nan
    S_true = data.S[te] if data.has_salinity else None

    preds = {"climatology": np.broadcast_to(climatology(data), true.shape).copy()}
    S_preds = {"climatology": None}
    cache = {}
    for rd in run_dirs:
        model, ck = load_model(os.path.join(rd, "best.pt"))
        h = ck.get("history", 0)
        if h not in cache:
            cache[h] = data_for(ck, data_path)
        T, S, _ = predict_all(model, cache[h], te)
        name = os.path.basename(os.path.normpath(rd))
        preds[name], S_preds[name] = T, S

    overall, tables = [], {}
    for name, p in preds.items():
        p = p.copy(); p[:, ~data.mask3d] = np.nan
        tables[name] = per_depth(true, p, data.depths)
        row = dict(model=name, **metrics(true, p))
        S_for_stab = S_preds[name] if S_preds[name] is not None else S_true
        row["density_inv_pct"] = 100 * instability_fraction(p, S_for_stab, data.mask3d)
        row["deep_T_inv_pct"] = deep_inversion_pct(p, data.depths, data.mask3d)
        row.update(product_rmse(p, true, data.depths))
        overall.append(row)
    overall = pd.DataFrame(overall)
    tgt_stab = 100 * instability_fraction(true, S_true, data.mask3d)
    tgt_deep = deep_inversion_pct(true, data.depths, data.mask3d)

    # wide per-depth RMSE table (one column per model) - the key comparison
    wide = pd.DataFrame({"depth_m": data.depths})
    for name, t in tables.items():
        wide[f"rmse_{name}"] = t["rmse"].values
        wide[f"bias_{name}"] = t["bias"].values
    overall.to_csv(os.path.join(out_dir, "glorys_test_overall.csv"), index=False)
    wide.to_csv(os.path.join(out_dir, "glorys_test_per_depth.csv"), index=False)
    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir, f"glorys_test_per_depth_{name}.csv"), index=False)

    print(f"\n=== Held-out TEST days ({len(te)} days) vs GLORYS ===")
    print(overall.round(3).to_string(index=False))
    print(f"(GLORYS target itself: density_inv {tgt_stab:.2f}%, deep_T_inv {tgt_deep:.2f}%)")
    print("\n=== Per-depth RMSE (deg C) ===")
    print(wide[["depth_m"] + [c for c in wide if c.startswith("rmse_")]].round(3).to_string(index=False))
    with open(os.path.join(out_dir, "glorys_test_summary.json"), "w") as f:
        json.dump({"test_days": [str(t)[:10] for t in data.times[te]],
                   "target_density_inv_pct": tgt_stab, "target_deep_T_inv_pct": tgt_deep,
                   "overall": overall.to_dict(orient="records")}, f, indent=1)
    return overall, wide


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    evaluate(a.data, a.runs, a.out)


if __name__ == "__main__":
    main()
