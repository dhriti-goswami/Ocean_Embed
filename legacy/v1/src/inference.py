
"""OceanEmbed inference wrapper -- what a backend/API calls."""
import torch
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from cnn import OceanCNN

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


class OceanEmbedPredictor:
    def __init__(self, model_path, norm_stats_path, hybrid_choice_path=None):
        self.model = OceanCNN(in_channels=5, out_depths=15)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        stats = np.load(norm_stats_path)
        self.x_mean, self.x_std = stats["x_mean"], stats["x_std"]
        self.y_mean, self.y_std = stats["y_mean"], stats["y_std"]

        self.hybrid_choice = None
        if hybrid_choice_path and os.path.exists(hybrid_choice_path):
            with open(hybrid_choice_path) as f:
                raw = json.load(f)
            self.hybrid_choice = {int(k): v for k, v in raw.items()}

    def predict(self, sss, ssh, wind_u, wind_v, bathymetry):
        X = np.stack([sss, ssh, wind_u, wind_v, bathymetry], axis=0)
        X_norm = (X - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        X_tensor = torch.tensor(X_norm[None], dtype=torch.float32)

        with torch.no_grad():
            pred_norm = self.model(X_tensor).numpy()[0]

        pred = pred_norm * self.y_std[:, None, None] + self.y_mean[:, None, None]

        if self.hybrid_choice is not None:
            for d, choice in self.hybrid_choice.items():
                if choice == "baseline":
                    pred[d] = self.y_mean[d]

        return pred

    def predict_at_point(self, sss, ssh, wind_u, wind_v, bathymetry, lat_idx, lon_idx):
        full = self.predict(sss, ssh, wind_u, wind_v, bathymetry)
        profile = full[:, lat_idx, lon_idx]
        return {depth: float(temp) for depth, temp in zip(TARGET_DEPTHS, profile)}
