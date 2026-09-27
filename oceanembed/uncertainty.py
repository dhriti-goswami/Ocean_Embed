"""Calibrated uncertainty ("confidence maps") from an ensemble of seeds.

For each seed: the heteroscedastic head gives sigma_i(x, z). Because training residuals
are smaller than unseen-day residuals, sigma_i is rescaled by a single factor k_i fitted on
the VALIDATION days (never the test days). The ensemble then combines
    mean  = average of member means
    var   = average of k_i^2 sigma_i^2  (what each model thinks it doesn't know)
          + variance of member means    (disagreement between models)
Calibration is checked on the TEST days against GLORYS and against independent ARGO floats:
a well-calibrated 90% interval should contain ~90% of the true values.

    python -m oceanembed.uncertainty --data cube.nc --runs runs/oceanembed_cyclone_s0 ... \
        --out results --outputs outputs --argo_cache results/argo_cache.csv
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .infer import load_model, predict_all, data_for, write_outputs
from .argo import fetch_argo, match

warnings.filterwarnings("ignore", category=RuntimeWarning)
LEVELS = np.array([0.5, 0.68, 0.8, 0.9, 0.95])


def coverage(err, std, levels=LEVELS):
    z = np.abs(err) / std
    m = np.isfinite(z)
    return {f"{int(round(l*100))}%": float((z[m] <= norm.ppf(0.5 + l / 2)).mean()) for l in levels}


def ensemble(run_dirs, data_path):
    members, info = [], []
    cache, data = {}, None
    for rd in run_dirs:
        model, ck = load_model(os.path.join(rd, "best.pt"))
        if not ck.get("uncertainty", False):
            raise ValueError(f"{rd} was trained without --uncertainty")
        h = ck.get("history", 0)
        if h not in cache:
            cache[h] = data_for(ck, data_path)
        data = cache[h]
        T, _, Z, sig = predict_all(model, data, return_sigma=True)
        va = data.idx["val"]
        true_v = data.T[va].copy(); true_v[:, ~data.mask3d] = np.nan
        zv = (true_v - T[va]) / sig[va]
        k = float(np.sqrt(np.nanmean(zv ** 2)))                    # fitted on validation days only
        members.append((T, sig * k, Z))
        info.append({"run": os.path.basename(os.path.normpath(rd)), "calibration_factor_k": k})
    mus = np.stack([m[0] for m in members])
    mean = mus.mean(0)
    var = np.mean(np.stack([m[1] ** 2 for m in members]), 0) + mus.var(0)
    return data, mean, np.sqrt(var), members[0][2], info


def reliability_plot(cov_rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for r in cov_rows:
        obs = [r[f"{int(round(l*100))}%"] for l in LEVELS]
        if r["subset"] == "all_days":
            continue
        ax.plot(LEVELS, obs, "o-", label=f"{r['against']} ({r['subset']})")
    ax.set_xlabel("nominal interval coverage"); ax.set_ylabel("observed coverage")
    ax.set_xlim(0.4, 1); ax.set_ylim(0.0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("Uncertainty calibration")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def run(data_path, run_dirs, out_dir, outputs_dir, argo_cache):
    os.makedirs(out_dir, exist_ok=True)
    data, mean, std, Z, info = ensemble(run_dirs, data_path)
    te = data.idx["test"]
    true = data.T.copy(); true[:, ~data.mask3d] = np.nan

    rows = []
    err = true[te] - mean[te]
    rows.append({"against": "GLORYS", "subset": "test_days", **coverage(err, std[te]),
                 "mean_std_degC": float(np.nanmean(std[te])),
                 "rmse_degC": float(np.sqrt(np.nanmean(err ** 2)))})
    # coverage by depth band (is the model more uncertain where it is less accurate?)
    band_rows = []
    for a, b in [(0, 10), (10, 50), (50, 100), (100, 200), (200, 500), (500, 950)]:
        k = (data.depths >= a) & (data.depths < b)
        e, s = err[:, k], std[te][:, k]
        band_rows.append({"band": f"{a}-{b} m", "rmse_degC": float(np.sqrt(np.nanmean(e ** 2))),
                          "mean_std_degC": float(np.nanmean(s)), **coverage(e, s)})

    # ARGO: interpolate ensemble mean and std to float measurement depths
    if True:  # ARGO calibration check
        df = fetch_argo(float(data.lon.min()), float(data.lon.max()), float(data.lat.min()),
                        float(data.lat.max()), str(data.times[0])[:10], str(data.times[-1])[:10],
                        cache=argo_cache)
        days = np.arange(len(data.times))
        mm = match(df, data, {d: mean[d] for d in days})
        ms = match(df, data, {d: std[d] for d in days})
        mm["std"] = ms["model"].values
        # Representativeness: a float measures a point, GLORYS a 1/12 deg model cell. That gap
        # (ARGO - GLORYS) is estimated per depth band on NON-test days and added in quadrature.
        mg = match(df, data, {d: true[d] for d in days})
        test_set = set(te.tolist())
        edges = [0, 10, 50, 100, 200, 500, 2000]
        fit = mg[~mg.day.isin(test_set)]
        rep = []
        for a, b in zip(edges[:-1], edges[1:]):
            r = fit[(fit.depth >= a) & (fit.depth < b)]
            rep.append(float(np.sqrt(np.mean((r.argo - r.model) ** 2))) if len(r) > 20 else np.nan)
        rep = np.array(rep)
        rep = np.where(np.isnan(rep), np.nanmean(rep), rep)
        band_idx = np.clip(np.searchsorted(edges, mm.depth.values, side="right") - 1, 0, len(rep) - 1)
        mm["std_repr"] = rep[band_idx]
        mm["std_total"] = np.sqrt(mm["std"] ** 2 + mm["std_repr"] ** 2)
        for subset, sel in [("all_days", mm), ("test_days", mm[mm.day.isin(test_set)])]:
            if len(sel):
                e = sel.argo.values - sel.model.values
                for kind, col in [("model sigma only", "std"), ("model + representativeness", "std_total")]:
                    rows.append({"against": f"ARGO [{kind}]", "subset": subset,
                                 **coverage(e, sel[col].values),
                                 "mean_std_degC": float(sel[col].mean()),
                                 "rmse_degC": float(np.sqrt(np.mean(e ** 2)))})
        with open(os.path.join(out_dir, "uncertainty_representativeness.json"), "w") as f:
            json.dump({"depth_edges_m": edges, "argo_minus_glorys_rms_degC": rep.tolist(),
                       "fitted_on": "non-test days"}, f, indent=1)
        mm.to_csv(os.path.join(out_dir, "uncertainty_argo_points.csv"), index=False)

    cov = pd.DataFrame(rows)
    bands = pd.DataFrame(band_rows)
    cov.to_csv(os.path.join(out_dir, "uncertainty_coverage.csv"), index=False)
    bands.to_csv(os.path.join(out_dir, "uncertainty_by_depth.csv"), index=False)
    reliability_plot(rows, os.path.join(out_dir, "fig_uncertainty_calibration.png"))
    with open(os.path.join(out_dir, "uncertainty_members.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("calibration factors (fitted on val days):", [round(i["calibration_factor_k"], 2) for i in info])
    print("\n=== Coverage: fraction of true values inside the predicted interval ===")
    print(cov.round(3).to_string(index=False))
    print("\n=== By depth (test days, GLORYS) ===")
    print(bands.round(3).to_string(index=False))

    if outputs_dir:
        mean_o, std_o = mean.astype(np.float32), std.astype(np.float32)
        p = write_outputs(data, mean_o, None, Z, outputs_dir,
                          meta=f"ensemble of {len(run_dirs)}: " + ", ".join(i["run"] for i in info),
                          T_std=std_o)
        print("wrote", *p)
    return cov, bands


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--outputs", default=None)
    p.add_argument("--argo_cache", default="data/argo_cache.csv")
    a = p.parse_args(argv)
    run(a.data, a.runs, a.out, a.outputs, a.argo_cache)


if __name__ == "__main__":
    main()
