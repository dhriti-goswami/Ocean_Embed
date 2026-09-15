
"""Hybrid depth-wise predictor for OceanEmbed."""
import numpy as np


def select_per_depth_method(val_true, val_pred, train_mean_per_depth, ocean_mask, target_depths):
    n_depth = val_true.shape[1]
    choice = {}
    print("Per-depth method selection (based on validation RMSE):")
    for d in range(n_depth):
        true_vals = val_true[:, d][:, ocean_mask]
        pred_vals = val_pred[:, d][:, ocean_mask]
        naive_vals = np.full_like(true_vals, train_mean_per_depth[d])

        cnn_rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
        naive_rmse = np.sqrt(np.mean((naive_vals - true_vals) ** 2))

        if cnn_rmse <= naive_rmse:
            choice[d] = "cnn"
        else:
            choice[d] = "baseline"

        print(f"  {int(target_depths[d]):>5}m  CNN RMSE: {cnn_rmse:.3f}  "
              f"baseline RMSE: {naive_rmse:.3f}  -> using: {choice[d]}")
    return choice


def hybrid_predict(cnn_pred, train_mean_per_depth, choice, lat_shape, lon_shape):
    n_time, n_depth = cnn_pred.shape[0], cnn_pred.shape[1]
    final = cnn_pred.copy()
    for d in range(n_depth):
        if choice[d] == "baseline":
            final[:, d] = train_mean_per_depth[d]
    return final
