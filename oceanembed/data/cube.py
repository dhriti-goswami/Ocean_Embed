"""Load the harmonized OceanEmbed data cube and turn it into model-ready tensors.

Expects the NetCDF produced by the team's data notebook
(`complete_model_ready_dataset.nc`): daily, 0.25 deg, Bay of Bengal, with surface
inputs on (time, lat, lon) and GLORYS `thetao` on (time, depth, lat, lon).
"""
from dataclasses import dataclass, field
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# Surface inputs (PS-specified products). ASCAT is left out on purpose: one swath per
# day spread by nearest-neighbour regridding is not a real gridded field, and CCMP
# already assimilates ASCAT winds.
DEFAULT_INPUTS = [
    "analysed_sst",  # OSTIA SST
    "sos",           # SMOS/SMAP SSS
    "sla",           # DUACS sea level anomaly
    "adt",           # DUACS absolute dynamic topography
    "u", "v",        # OSCAR surface currents
    "uwnd", "vwnd",  # CCMP 10 m winds
]
TARGET_T = "thetao"
TARGET_S = "so"


@dataclass
class DataConfig:
    path: str
    inputs: list = field(default_factory=lambda: list(DEFAULT_INPUTS))
    use_salinity_target: bool = True     # used only if `so` exists in the file
    train_end: str = "2023-07-31"
    val_end: str = "2023-08-15"          # test = everything after val_end
    pad_multiple: int = 8
    min_input_coverage: float = 0.9      # fraction of days an input must be valid at a cell
    mld_threshold: float = 0.5           # deg C drop from the 10 m reference
    history_days: int = 2                # also feed surface fields from t-1..t-k (ocean memory)


def _standardize(ds):
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.variables}
    ds = ds.rename(ren) if ren else ds
    return ds.sortby("lat").sortby("lon")


def compute_mld(T, depths, threshold=0.5):
    """Temperature-criterion mixed layer depth. T: (N, D, H, W) deg C with NaN below seabed."""
    k10 = int(np.argmin(np.abs(depths - 10.0)))
    ref = T[:, k10:k10 + 1]
    below = (T < ref - threshold) & (np.arange(len(depths))[None, :, None, None] > k10)
    first = np.where(below.any(1), below.argmax(1), -1)
    deepest_valid = (~np.isnan(T)).sum(1) - 1
    idx = np.where(first >= 0, first, np.clip(deepest_valid, 0, None))
    return depths[idx].astype(np.float32)          # (N, H, W)


class OceanData:
    """Holds the whole cube in memory (it is small: 92 x 69 x 81)."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        ds = _standardize(xr.open_dataset(cfg.path))
        missing = [v for v in cfg.inputs + [TARGET_T] if v not in ds]
        if missing:
            raise KeyError(f"Missing variables in cube: {missing}. Found: {list(ds.data_vars)}")

        self.times = ds.time.values
        self.lat = ds.lat.values.astype(np.float32)
        self.lon = ds.lon.values.astype(np.float32)
        self.depths = ds.depth.values.astype(np.float32)

        # ---- inputs (N, C, H, W) ----
        chans, names = [], []
        for v in cfg.inputs:
            a = ds[v].squeeze(drop=True).transpose("time", "lat", "lon").values.astype(np.float32)
            if v == "analysed_sst" and np.nanmax(a) > 100:
                a = a - 273.15                                  # Kelvin -> Celsius
            chans.append(a); names.append(v)
        n, h, w = chans[0].shape
        doy = ds.time.dt.dayofyear.values.astype(np.float32)
        chans.append(np.broadcast_to(np.sin(2 * np.pi * doy / 365.25)[:, None, None], (n, h, w)))
        chans.append(np.broadcast_to(np.cos(2 * np.pi * doy / 365.25)[:, None, None], (n, h, w)))
        chans.append(np.broadcast_to(self.lat[None, :, None], (n, h, w)))
        chans.append(np.broadcast_to(self.lon[None, None, :], (n, h, w)))
        names += ["doy_sin", "doy_cos", "lat", "lon"]
        X = np.stack(chans, 1).astype(np.float32)
        self.channel_names = names

        # ---- targets (N, D, H, W) ----
        T = ds[TARGET_T].transpose("time", "depth", "lat", "lon").values.astype(np.float32)
        self.has_salinity = cfg.use_salinity_target and TARGET_S in ds
        S = (ds[TARGET_S].transpose("time", "depth", "lat", "lon").values.astype(np.float32)
             if self.has_salinity else None)
        ds.close()

        # ---- masks ----
        # 3D target mask: a level is ocean if valid on every day (seabed cuts it off on shelves)
        m3 = (~np.isnan(T)).all(0)
        if S is not None:
            m3 &= (~np.isnan(S)).all(0)
        n_in = len(cfg.inputs)
        cover = (~np.isnan(X[:, :n_in])).mean(0)                 # (C, H, W)
        m2 = m3[0] & (cover >= cfg.min_input_coverage).all(0)
        m3 &= m2[None]
        self.mask2d, self.mask3d = m2, m3

        # ---- splits (contiguous in time: no leakage between neighbouring days) ----
        t = self.times
        self.idx = {
            "train": np.where(t <= np.datetime64(cfg.train_end))[0],
            "val": np.where((t > np.datetime64(cfg.train_end)) & (t <= np.datetime64(cfg.val_end)))[0],
            "test": np.where(t > np.datetime64(cfg.val_end))[0],
        }
        if min(len(v) for v in self.idx.values()) == 0:      # dates outside the cube: fall back
            n_tr, n_va = int(0.66 * n), int(0.17 * n)
            self.idx = {"train": np.arange(n_tr), "val": np.arange(n_tr, n_tr + n_va),
                        "test": np.arange(n_tr + n_va, n)}

        # ---- normalization from TRAIN days only ----
        tr = self.idx["train"]
        self.x_mean = np.array([np.nanmean(X[tr, c][:, m2]) for c in range(X.shape[1])], np.float32)
        self.x_std = np.array([np.nanstd(X[tr, c][:, m2]) + 1e-6 for c in range(X.shape[1])], np.float32)
        self.t_mean = np.array([np.nanmean(T[tr, d][:, m3[d]]) if m3[d].any() else 0.0
                                for d in range(len(self.depths))], np.float32)
        self.t_std = np.array([np.nanstd(T[tr, d][:, m3[d]]) + 1e-3 if m3[d].any() else 1.0
                               for d in range(len(self.depths))], np.float32)
        if S is not None:
            self.s_mean = np.array([np.nanmean(S[tr, d][:, m3[d]]) if m3[d].any() else 35.0
                                    for d in range(len(self.depths))], np.float32)
            self.s_std = np.array([np.nanstd(S[tr, d][:, m3[d]]) + 1e-3 if m3[d].any() else 1.0
                                   for d in range(len(self.depths))], np.float32)

        Xn = (X - self.x_mean[None, :, None, None]) / self.x_std[None, :, None, None]
        Xn[:, :, ~m2] = 0.0
        Xn = np.nan_to_num(Xn, nan=0.0)
        # Temporal context: surface fields from previous days (subsurface responds with a lag).
        # Only past/present surface observations are used - never future days, never targets.
        k = cfg.history_days
        if k > 0:
            dyn, stat = Xn[:, :n_in], Xn[:, n_in:]
            lagged = [dyn[np.clip(np.arange(n) - lag, 0, None)] for lag in range(k, 0, -1)]
            Xn = np.concatenate(lagged + [dyn, stat], 1)
            base = self.channel_names
            self.channel_names = ([f"{v}_t-{lag}" for lag in range(k, 0, -1) for v in base[:n_in]]
                                  + base)
        self.X = np.ascontiguousarray(Xn, dtype=np.float32)
        self.T = T                                                 # deg C, NaN kept (for eval)
        self.S = S
        self.mld = compute_mld(T, self.depths, cfg.mld_threshold)  # (N, H, W)

        # ---- padding so the U-Net / ViT patch grid divides evenly ----
        pm = cfg.pad_multiple
        self.H, self.W = h, w
        self.Hp, self.Wp = -(-h // pm) * pm, -(-w // pm) * pm

    # ------------------------------------------------------------------
    def pad(self, a):
        ph, pw = self.Hp - self.H, self.Wp - self.W
        return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)])

    def unpad(self, a):
        return a[..., : self.H, : self.W]

    def norm_stats(self):
        d = dict(x_mean=self.x_mean, x_std=self.x_std, t_mean=self.t_mean, t_std=self.t_std)
        if self.has_salinity:
            d.update(s_mean=self.s_mean, s_std=self.s_std)
        return d

    def dataset(self, split):
        return OceanTorchDataset(self, self.idx[split])

    @property
    def n_inputs(self):
        return self.X.shape[1]

    @property
    def sla_channel(self):
        return self.channel_names.index("sla") if "sla" in self.channel_names else None


class OceanTorchDataset(Dataset):
    def __init__(self, data: OceanData, indices):
        self.d, self.ix = data, np.asarray(indices)
        m3 = data.pad(data.mask3d)
        self.m3 = torch.from_numpy(m3)

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        d, k = self.d, self.ix[i]
        x = d.pad(d.X[k])
        tn = (d.T[k] - d.t_mean[:, None, None]) / d.t_std[:, None, None]
        y = [np.nan_to_num(tn, nan=0.0)]
        if d.has_salinity:
            sn = (d.S[k] - d.s_mean[:, None, None]) / d.s_std[:, None, None]
            y.append(np.nan_to_num(sn, nan=0.0))
        y = d.pad(np.concatenate(y, 0).astype(np.float32))
        mld = d.pad(d.mld[k][None])[0]
        return (torch.from_numpy(x), torch.from_numpy(y), self.m3,
                torch.from_numpy(mld.astype(np.float32)), int(k))
