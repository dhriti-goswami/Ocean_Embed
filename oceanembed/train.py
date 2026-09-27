"""Train one model variant.

    python -m oceanembed.train --data cube.nc --out runs/oceanembed_pinn --variant oceanembed --physics
    python -m oceanembed.train --data cube.nc --out runs/oceanembed_nophys --variant oceanembed
    python -m oceanembed.train --data cube.nc --out runs/unet --variant unet
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DataConfig, OceanData
from .models import build_model
from .physics import PhysicsLoss, reference_density_profile


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
    data = OceanData(DataConfig(path=args.data))
    D = len(data.depths)
    out_vars = 2 if data.has_salinity else 1
    print(f"device={device}  grid={data.H}x{data.W} (padded {data.Hp}x{data.Wp})  depths={D}  "
          f"inputs={data.channel_names}  salinity_target={data.has_salinity}")
    print({k: len(v) for k, v in data.idx.items()}, "days per split")

    tr = DataLoader(data.dataset("train"), batch_size=args.batch, shuffle=True, drop_last=False)
    va = DataLoader(data.dataset("val"), batch_size=args.batch)

    model = build_model(args.variant, data.n_inputs, D, (data.Hp, data.Wp), out_vars).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant={args.variant} physics={args.physics} params={n_params/1e6:.2f}M")

    phys = None
    if args.physics:
        phys = PhysicsLoss(data.depths, data.norm_stats(), reference_density_profile(data),
                           data.has_salinity, data.sla_channel,
                           w_mld=args.w_mld, w_stab=args.w_stab, w_steric=args.w_steric).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out, exist_ok=True)
    best, best_ep, history = float("inf"), -1, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        # physics weight ramps in after a warm-up so the net first learns the data
        ramp = 0.0 if ep <= args.warmup else min(1.0, (ep - args.warmup) / max(args.warmup, 1))
        agg = {"data": 0.0, "total": 0.0}
        for x, y, m3, mld, _ in tr:
            x, y, m3, mld = x.to(device), y.to(device), m3.to(device), mld.to(device)
            out = model(x)
            loss, lt = masked_mse(out, y, m3, D)
            agg["data"] += lt
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
        if v_rmse < best:
            best, best_ep = v_rmse, ep
            torch.save({"model": model.state_dict(), "variant": args.variant, "out_vars": out_vars,
                        "in_ch": data.n_inputs, "grid": (data.Hp, data.Wp), "depths": data.depths,
                        "channels": data.channel_names, "norm": data.norm_stats(),
                        "physics": args.physics, "epoch": ep, "val_rmse_degC": v_rmse},
                       os.path.join(args.out, "best.pt"))
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            extra = " ".join(f"{k}={v:.4f}" for k, v in agg.items() if k not in ("data", "total"))
            print(f"ep {ep:4d} | data {agg['data']:.4f} | val RMSE {v_rmse:.3f} C | "
                  f"best {best:.3f} @ {best_ep} | {extra} | {time.time()-t0:.0f}s")
        if ep - best_ep >= args.patience:
            print(f"early stop at epoch {ep} (no val improvement for {args.patience} epochs)")
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    summary = {"variant": args.variant, "physics": args.physics, "best_val_rmse_degC": best,
               "best_epoch": best_ep, "params": n_params, "train_seconds": time.time() - t0}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done:", summary)
    return summary


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="oceanembed", choices=["oceanembed", "unet"])
    p.add_argument("--physics", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--w_mld", type=float, default=0.05)
    p.add_argument("--w_stab", type=float, default=1.0)
    p.add_argument("--w_steric", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
