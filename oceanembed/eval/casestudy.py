"""Case study: reconstructing the upper-ocean response to the deep depression over the
north-east Bay of Bengal (IMD: depression -> deep depression on 1 Aug 2023, centred near
21.2 N, 91.2 E, crossing the Bangladesh coast the same evening).

Compares, for a box over the north-east Bay, daily box-mean SST (satellite input) and the
reconstructed MLD, D26 and TCHP against GLORYS and against ARGO floats in the box.

    python -m oceanembed casestudy --data cube.nc --runs runs/oceanembed_cyclone_s0 ... --out results
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from oceanembed.inference import load_model, predict_all, data_for
from oceanembed.physics.products import all_products
from oceanembed.eval.argo import fetch_argo, argo_products

warnings.filterwarnings("ignore", category=RuntimeWarning)
EVENT = {"name": "Deep depression, NE Bay of Bengal", "date": "2023-08-01", "lat": 21.2, "lon": 91.2}
BOX = dict(lat=(18.5, 22.0), lon=(87.5, 92.5))
WINDOW = ("2023-07-18", "2023-08-15")
Q = ["MLD_m", "D26_m", "TCHP_kJcm2"]
QLAB = {"MLD_m": "Mixed layer depth (m)", "D26_m": "26 °C isotherm depth (m)",
        "TCHP_kJcm2": "Cyclone heat potential (kJ/cm²)"}


def box_mean(field, data, box_mask):
    return np.array([np.nanmean(f[box_mask]) for f in field])


def run(data_path, run_dirs, out_dir, argo_cache=None, before="2023-07-28", after="2023-08-04"):
    os.makedirs(out_dir, exist_ok=True)
    members, data, cache = [], None, {}
    for rd in run_dirs:
        model, ck = load_model(os.path.join(rd, "best.pt"))
        h = ck.get("history", 0)
        if h not in cache:
            cache[h] = data_for(ck, data_path)
        data = cache[h]
        T, _, _ = predict_all(model, data)
        members.append(all_products(T, data.depths))
    true = data.T.copy(); true[:, ~data.mask3d] = np.nan
    P_true = all_products(true, data.depths)

    LA, LO = np.meshgrid(data.lat, data.lon, indexing="ij")
    box = (LA >= BOX["lat"][0]) & (LA <= BOX["lat"][1]) & (LO >= BOX["lon"][0]) & (LO <= BOX["lon"][1]) \
          & data.mask2d & data.mask3d[20]                  # ocean deeper than ~60 m
    t = pd.to_datetime(data.times)
    win = (t >= WINDOW[0]) & (t <= WINDOW[1])

    ds = xr.open_dataset(data_path)
    sst = ds["analysed_sst"].sortby("lat").sortby("lon").transpose("time", "lat", "lon").values
    if np.nanmax(sst) > 100:
        sst = sst - 273.15
    ds.close()

    rows = {"date": t[win]}
    rows["SST_satellite"] = box_mean(sst, data, box)[win]
    for q in Q:
        rows[f"{q}_glorys"] = box_mean(P_true[q], data, box)[win]
        ms = np.stack([box_mean(m[q], data, box) for m in members])[:, win]
        rows[f"{q}_model"] = ms.mean(0)
        rows[f"{q}_model_spread"] = ms.std(0)
    ts = pd.DataFrame(rows)
    ts.to_csv(os.path.join(out_dir, "casestudy_timeseries.csv"), index=False)

    argo_box = None
    try:
        df = fetch_argo(float(data.lon.min()), float(data.lon.max()), float(data.lat.min()),
                        float(data.lat.max()), str(data.times[0])[:10], str(data.times[-1])[:10],
                        cache=argo_cache)
        df = df[(df.latitude.between(*BOX["lat"])) & (df.longitude.between(*BOX["lon"]))]
        if len(df):
            model, ck = load_model(os.path.join(run_dirs[0], "best.pt"))
            T0, _, _ = predict_all(model, cache[ck.get("history", 0)])
            argo_box = argo_products(df, data, {d: T0[d] for d in range(len(data.times))}, min_bottom=150)
            if len(argo_box):
                argo_box["date"] = t[argo_box.day.values]
                argo_box.to_csv(os.path.join(out_dir, "casestudy_argo.csv"), index=False)
    except Exception as e:
        print("ARGO for case study skipped:", e)

    # ---- figure 1: time series ----
    ev = pd.Timestamp(EVENT["date"])
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(ts.date, ts.SST_satellite, "C3-", label="satellite SST (OSTIA, input)")
    axes[0].set_ylabel("SST (°C)")
    for ax, q in zip(axes[1:], Q):
        ax.plot(ts.date, ts[f"{q}_glorys"], "k-", lw=2, label="GLORYS (reference)")
        ax.plot(ts.date, ts[f"{q}_model"], "C0-", label="OceanEmbed (satellite only)")
        ax.fill_between(ts.date, ts[f"{q}_model"] - ts[f"{q}_model_spread"],
                        ts[f"{q}_model"] + ts[f"{q}_model_spread"], color="C0", alpha=0.25)
        if argo_box is not None and len(argo_box):
            ax.plot(argo_box.date, argo_box[f"argo_{q}"], "o", color="C2", ms=5, label="ARGO floats")
        ax.set_ylabel(QLAB[q], fontsize=8)
    for ax in axes:
        ax.axvline(ev, color="grey", ls="--"); ax.grid(alpha=0.3)
        ax.axvspan(pd.Timestamp(WINDOW[0]), pd.Timestamp("2023-07-31 23:59"), color="0.9", zorder=0)
    axes[0].text(ev, axes[0].get_ylim()[1], " deep depression", va="top", fontsize=8)
    axes[1].legend(fontsize=7, loc="best")
    axes[0].set_title(f"{EVENT['name']} — box {BOX['lat'][0]}–{BOX['lat'][1]}°N, "
                      f"{BOX['lon'][0]}–{BOX['lon'][1]}°E\n(grey shading = training days)", fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout()
    f1 = os.path.join(out_dir, "fig_casestudy_timeseries.png"); fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- figure 2: TCHP change map (after - before), model vs GLORYS ----
    ib, ia = int(np.where(t == pd.Timestamp(before))[0][0]), int(np.where(t == pd.Timestamp(after))[0][0])
    dm = np.mean([m["TCHP_kJcm2"][ia] - m["TCHP_kJcm2"][ib] for m in members], 0)
    dg = P_true["TCHP_kJcm2"][ia] - P_true["TCHP_kJcm2"][ib]
    lim = np.nanpercentile(np.abs(np.concatenate([dm[data.mask2d], dg[data.mask2d]])), 98)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, f, ttl in [(axes[0], dg, "GLORYS (reference)"), (axes[1], dm, "OceanEmbed (satellite only)")]:
        im = ax.pcolormesh(data.lon, data.lat, np.where(data.mask2d, f, np.nan), cmap="RdBu_r",
                           vmin=-lim, vmax=lim, shading="auto")
        ax.plot(EVENT["lon"], EVENT["lat"], "k*", ms=12)
        ax.add_patch(plt.Rectangle((BOX["lon"][0], BOX["lat"][0]), BOX["lon"][1] - BOX["lon"][0],
                                   BOX["lat"][1] - BOX["lat"][0], fill=False, ls="--"))
        ax.set_title(ttl, fontsize=10); ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)", fontsize=8); ax.set_ylabel("Latitude (°N)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="ΔTCHP (kJ/cm²)")
    fig.suptitle(f"Change in cyclone heat potential, {after} minus {before} "
                 f"(★ = deep depression, 1 Aug 2023)", fontsize=10)
    f2 = os.path.join(out_dir, "fig_casestudy_tchp_change.png"); fig.savefig(f2, dpi=150); plt.close(fig)

    # correlation of daily box-mean evolution, model vs GLORYS
    stats = {q: {"corr_daily": float(np.corrcoef(ts[f"{q}_glorys"], ts[f"{q}_model"])[0, 1]),
                 "rmse": float(np.sqrt(np.mean((ts[f"{q}_glorys"] - ts[f"{q}_model"]) ** 2)))} for q in Q}
    pd.DataFrame(stats).T.to_csv(os.path.join(out_dir, "casestudy_stats.csv"))
    print("box cells:", int(box.sum()))
    print(ts.round(2).to_string(index=False))
    print("\nmodel vs GLORYS over the window:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in stats.items()})
    if argo_box is not None:
        print(f"ARGO profiles in box: {len(argo_box)}")
    return f1, f2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--argo_cache", default=None)
    a = p.parse_args(argv)
    run(a.data, a.runs, a.out, a.argo_cache)


if __name__ == "__main__":
    main()
