"""Train one model.

    python -m oceanembed train --config configs/main.yaml --data cube.nc --out runs/main_s0 --seed 0

Configs: configs/main.yaml (main model), configs/ablation_*.yaml, configs/smoke.yaml.
Any config value can be overridden on the command line.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from oceanembed.data.cube import DataConfig, OceanData
from oceanembed.models.network import build_model
from oceanembed.physics.losses import PhysicsLoss, reference_density_profile
from oceanembed.config import apply_config


def gaussian_nll(logvar, mu, y, m3):
    """Heteroscedastic NLL for the temperature channels. The mean is detached, so learning the
    uncertainty never degrades the mean prediction; the head learns where errors are large."""
    s = logvar.clamp(-10, 6)
    mt = m3.float()
    nll = 0.5 * (torch.exp(-s) * (y - mu.detach()).pow(2) + s)
    return (nll * mt).sum() / mt.sum().clamp_min(1)


def masked_mse(out, y, m3, n_depth, s_weight=0.5):
    """MSE over valid ocean levels; temperature channels + (down-weighted) salinity channels."""
    mt = m3.float()
    lt = ((out[:, :n_depth] - y[:, :n_depth]).pow(2) * mt).sum() / mt.sum().clamp_min(1)
    if out.shape[1] > n_depth:
        ls = ((out[:, n_depth:] - y[:, n_depth:]).pow(2) * mt).sum() / mt.sum().clamp_min(1)
        return lt + s_weight * ls, float(lt.detach())
    return lt, float(lt.detach())


@torch.no_grad()
def val_rmse_degC(model, loader, data, device):
    """Temperature RMSE in deg C over all valid ocean levels."""
    model.eval()
    D = len(data.depths)
    std = torch.as_tensor(data.t_std, device=device)[None, :, None, None]
    se, n = 0.0, 0
    for x, y, m3, _, _ in loader:
        x, y, m3 = x.to(device), y.to(device), m3.to(device)
        diff = (model(x)[:, :D] - y[:, :D]) * std
        se += float((diff.pow(2) * m3).sum()); n += int(m3.sum())
    return (se / max(n, 1)) ** 0.5


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = OceanData(DataConfig(path=args.data, history_days=args.history))
    D = len(data.depths)
    out_vars = 2 if data.has_salinity else 1
    print(f"device={device}  grid={data.H}x{data.W} (padded {data.Hp}x{data.Wp})  depths={D}  "
          f"inputs={data.channel_names}  salinity_target={data.has_salinity}")
    print({k: len(v) for k, v in data.idx.items()}, "days per split")

    tr = DataLoader(data.dataset("train"), batch_size=args.batch, shuffle=True, drop_last=False)
    va = DataLoader(data.dataset("val"), batch_size=args.batch)

    model = build_model(args.variant, data.n_inputs, D, (data.Hp, data.Wp), out_vars,
                        uncertainty=args.uncertainty).to(device)
    K = D * out_vars                                   # number of mean channels
    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant={args.variant} physics={args.physics} params={n_params/1e6:.2f}M")

    phys = None
    if args.physics:
        phys = PhysicsLoss(data.depths, data.norm_stats(), reference_density_profile(data),
                           data.has_salinity, data.sla_channel,
                           w_mld=args.w_mld, w_stab=args.w_stab, w_steric=args.w_steric,
                           mode=args.phys_mode, w_quant=args.w_quant).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out, exist_ok=True)
    best, best_ep, history = float("inf"), -1, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        # physics weight ramps in after a warm-up so the net first learns the data
        ramp = 0.0 if ep <= args.warmup else min(1.0, (ep - args.warmup) / max(args.ramp, 1))
        agg = {"data": 0.0, "total": 0.0}
        for x, y, m3, mld, _ in tr:
            x, y, m3, mld = x.to(device), y.to(device), m3.to(device), mld.to(device)
            if args.noise > 0:                       # input-noise augmentation (61 days is tiny)
                ocean = m3[:, :1].float()
                x = x + args.noise * torch.randn_like(x) * ocean
            full = model(x)
            out = full[:, :K]
            loss, lt = masked_mse(out, y, m3, D)
            agg["data"] += lt
            if args.uncertainty:
                ln = gaussian_nll(full[:, K:K + D], out[:, :D], y[:, :D], m3)
                loss = loss + 0.1 * ln
                agg["nll"] = agg.get("nll", 0.0) + float(ln.detach())
            if phys is not None and ramp > 0:
                lp, terms = phys(out, y, m3, mld, x)
                loss = loss + ramp * lp
                for k, v in terms.items():
                    agg[k] = agg.get(k, 0.0) + v
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            agg["total"] += float(loss.detach())
        sched.step()
        agg = {k: v / len(tr) for k, v in agg.items()}
        v_rmse = val_rmse_degC(model, va, data, device)
        history.append({"epoch": ep, "val_rmse_degC": v_rmse, "physics_ramp": ramp, **agg})
        select_from = min(args.warmup + args.ramp, args.epochs) if phys is not None else 1
        if ep >= select_from and v_rmse < best:
            best, best_ep = v_rmse, ep
            torch.save({"model": model.state_dict(), "variant": args.variant, "out_vars": out_vars,
                        "uncertainty": args.uncertainty, "history": args.history,
                        "phys_mode": args.phys_mode if args.physics else None,
                        "in_ch": data.n_inputs, "grid": (data.Hp, data.Wp), "depths": data.depths,
                        "channels": data.channel_names, "norm": data.norm_stats(),
                        "physics": args.physics, "epoch": ep, "val_rmse_degC": v_rmse},
                       os.path.join(args.out, "best.pt"))
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            extra = " ".join(f"{k}={v:.4f}" for k, v in agg.items() if k not in ("data", "total"))
            print(f"ep {ep:4d} | data {agg['data']:.4f} | val RMSE {v_rmse:.3f} C | "
                  f"best {best:.3f} @ {best_ep} | {extra} | {time.time()-t0:.0f}s")
        if best_ep > 0 and ep - best_ep >= args.patience:
            print(f"early stop at epoch {ep} (no val improvement for {args.patience} epochs)")
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    summary = {"variant": args.variant, "physics": args.physics,
               "phys_mode": args.phys_mode if args.physics else None,
               "uncertainty": args.uncertainty, "history": args.history, "best_val_rmse_degC": best,
               "best_epoch": best_ep, "params": n_params, "train_seconds": time.time() - t0}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done:", summary)
    return summary


def parse(argv=None):
    p = argparse.ArgumentParser(prog="python -m oceanembed train")
    p.add_argument("--config", default=None, help="YAML file of defaults (configs/*.yaml)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="oceanembed", choices=["oceanembed", "unet"])
    p.add_argument("--physics", action="store_true")
    p.add_argument("--phys_mode", default="cyclone", choices=["cyclone", "constraints"])
    p.add_argument("--w_quant", type=float, default=0.1)
    p.add_argument("--uncertainty", action="store_true")
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--ramp", type=int, default=5)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--w_mld", type=float, default=0.05)
    p.add_argument("--w_stab", type=float, default=1.0)
    p.add_argument("--w_steric", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return apply_config(p, argv)


if __name__ == "__main__":
    train(parse())
