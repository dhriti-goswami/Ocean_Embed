
"""Evaluation metrics for OceanEmbed."""
import numpy as np


def denormalize(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def compute_metrics(y_true, y_pred, mask):
    n_depth = y_true.shape[1]
    overall = _metrics_flat(y_true[:, :, mask].ravel(), y_pred[:, :, mask].ravel())
    per_depth = []
    for d in range(n_depth):
        t = y_true[:, d][:, mask].ravel()
        p = y_pred[:, d][:, mask].ravel()
        per_depth.append(_metrics_flat(t, p))
    return {"overall": overall, "per_depth": per_depth}


def _metrics_flat(t, p):
    diff = p - t
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if t.std() > 0 and p.std() > 0:
        corr = np.corrcoef(t, p)[0, 1]
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "correlation": corr, "bias": bias, "n": len(t)}


def print_metrics_report(metrics, target_depths):
    o = metrics["overall"]
    rmse = o["rmse"]; mae = o["mae"]; r2 = o["r2"]; corr = o["correlation"]; bias = o["bias"]; n = o["n"]
    print("=== Overall ===")
    print("  RMSE:", round(rmse, 3), " MAE:", round(mae, 3), " R2:", round(r2, 3),
          " Corr:", round(corr, 3), " Bias:", round(bias, 3), " n:", n)
    print()
    print("=== Per depth ===")
    for depth, m in zip(target_depths, metrics["per_depth"]):
        print("  ", int(depth), "m  RMSE:", round(m["rmse"], 3), " MAE:", round(m["mae"], 3),
              " R2:", round(m["r2"], 3), " Corr:", round(m["correlation"], 3), " Bias:", round(m["bias"], 3))
