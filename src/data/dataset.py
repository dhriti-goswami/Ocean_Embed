
"""PyTorch Dataset for OceanEmbed."""
import numpy as np
import torch
from torch.utils.data import Dataset


class OceanDataset(Dataset):
    def __init__(self, X, Y, ocean_mask, x_mean=None, x_std=None, y_mean=None, y_std=None):
        X_filled = np.nan_to_num(X, nan=0.0)
        Y_filled = np.nan_to_num(Y, nan=0.0)

        if x_mean is None:
            x_mean = np.array([X[:, c][:, ocean_mask].mean() for c in range(X.shape[1])])
            x_std = np.array([X[:, c][:, ocean_mask].std() for c in range(X.shape[1])])
            y_mean = Y[:, :, ocean_mask].mean()
            y_std = Y[:, :, ocean_mask].std()

        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        X_norm = (X_filled - x_mean[None, :, None, None]) / x_std[None, :, None, None]
        Y_norm = (Y_filled - y_mean) / y_std

        self.X = torch.tensor(X_norm, dtype=torch.float32)
        self.Y = torch.tensor(Y_norm, dtype=torch.float32)
        self.mask = torch.tensor(ocean_mask, dtype=torch.bool)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask

    def get_norm_stats(self):
        return self.x_mean, self.x_std, self.y_mean, self.y_std
