"""Demo / report figures. All functions save a PNG and return its path."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def per_depth_rmse(csv_path, out_png):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.5, 7))
    for c in [c for c in df.columns if c.startswith("rmse_")]:
        ax.plot(df[c], df.depth_m, marker="o", ms=3, label=c.replace("rmse_", ""))
    ax.invert_yaxis(); ax.set_yscale("symlog", linthresh=20)
    ax.set_xlabel("RMSE vs GLORYS, held-out days (°C)"); ax.set_ylabel("Depth (m)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Error by depth")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def training_curves(run_dirs, out_png):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rd in run_dirs:
        h = json.load(open(os.path.join(rd, "history.json")))
        ax.plot([r["epoch"] for r in h], [r["val_rmse_degC"] for r in h],
                label=os.path.basename(os.path.normpath(rd)))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation RMSE (°C)"); ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(); ax.set_title("Training convergence")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def argo_profiles(matches_csv, out_png, n=6, seed=0):
    """Model vs ARGO for a few random floats (the most intuitive validation visual)."""
    m = pd.read_csv(matches_csv)
    groups = list(m.groupby(["platform", "day"]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(groups), size=min(n, len(groups)), replace=False)
    fig, axes = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, pick):
        (plat, day), g = groups[i]
        ax.plot(g.argo, g.depth, "k.", ms=3, label="ARGO")
        ax.plot(g.model, g.depth, "r-", lw=1.5, label="OceanEmbed")
        ax.set_title(f"float {plat}\nday {day}", fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlabel("°C")
    axes[0].invert_yaxis(); axes[0].set_ylabel("Depth (m)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def surface_vs_subsurface(nc_path, day, out_png, depths=(0, 50, 100, 200, 500)):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    t = ds.temperature.isel(time=day)
    fig, axes = plt.subplots(1, len(depths), figsize=(3.2 * len(depths), 3.4))
    for ax, d in zip(axes, depths):
        layer = t.sel(depth=d, method="nearest")
        im = ax.pcolormesh(ds.longitude, ds.latitude, layer, cmap="RdYlBu_r", shading="auto")
        ax.set_title(f"{float(layer.depth):.0f} m", fontsize=9); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8, label="°C")
    fig.suptitle(f"Reconstructed temperature, {str(ds.time.values[day])[:10]}")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png
