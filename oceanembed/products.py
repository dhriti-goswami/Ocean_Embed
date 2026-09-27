"""Operational upper-ocean products derived from temperature profiles.

  D26  - depth of the 26 deg C isotherm (m): depth of water warm enough to fuel cyclones
  D20  - depth of the 20 deg C isotherm (m): thermocline depth proxy
  TCHP - Tropical Cyclone Heat Potential (kJ cm^-2): rho*cp * integral of (T - 26) above D26
  MLD  - mixed layer depth (m): 0.5 deg C drop from the 10 m temperature

Works on any array whose depth axis is `axis` (default 1 for (N, D, H, W)); NaN = no data.
"""
import numpy as np

RHO0, CP = 1025.0, 3990.0


def _move(T, axis):
    return np.moveaxis(np.asarray(T, dtype=np.float64), axis, -1)        # (..., D)


def isotherm_depth(T, depths, iso, axis=1):
    """Linear interpolation of the first downward crossing of `iso`.
    0 if the surface is already colder; NaN if the column never gets colder (censored)."""
    Tm = _move(T, axis)
    z = np.asarray(depths, np.float64)
    below = Tm < iso
    has = below[..., 1:] & ~below[..., :-1]                               # crossing between k-1,k
    k = np.where(has.any(-1), has.argmax(-1) + 1, -1)
    kk = np.clip(k, 1, None)
    t0 = np.take_along_axis(Tm, (kk - 1)[..., None], -1)[..., 0]
    t1 = np.take_along_axis(Tm, kk[..., None], -1)[..., 0]
    frac = np.clip((t0 - iso) / np.where(t0 - t1 == 0, np.nan, t0 - t1), 0, 1)
    d = z[kk - 1] + frac * (z[kk] - z[kk - 1])
    d = np.where(k > 0, d, np.nan)
    d = np.where(Tm[..., 0] < iso, 0.0, d)
    return np.where(np.isnan(Tm[..., 0]), np.nan, d)


def layer_thickness(depths):
    d = np.asarray(depths, np.float64)
    edges = np.concatenate([[0.0], 0.5 * (d[1:] + d[:-1]), [d[-1] + 0.5 * (d[-1] - d[-2])]])
    return np.diff(edges)


def tchp(T, depths, axis=1):
    """kJ/cm^2. Integral of rho*cp*(T-26)+ dz over the valid column."""
    Tm = _move(T, axis)
    dz = layer_thickness(depths)
    q = RHO0 * CP * np.nansum(np.clip(Tm - 26.0, 0, None) * dz, -1) / 1e7
    return np.where(np.isnan(Tm[..., 0]), np.nan, q)


def mld(T, depths, axis=1, threshold=0.5):
    Tm = _move(T, axis)
    z = np.asarray(depths, np.float64)
    k10 = int(np.argmin(np.abs(z - 10.0)))
    ref = Tm[..., k10:k10 + 1]
    below = (Tm < ref - threshold)
    below[..., : k10 + 1] = False
    k = np.where(below.any(-1), below.argmax(-1), -1)
    kk = np.clip(k, 1, None)
    tgt = ref[..., 0] - threshold
    t0 = np.take_along_axis(Tm, (kk - 1)[..., None], -1)[..., 0]
    t1 = np.take_along_axis(Tm, kk[..., None], -1)[..., 0]
    frac = np.clip((t0 - tgt) / np.where(t0 - t1 == 0, np.nan, t0 - t1), 0, 1)
    d = np.where(k > 0, z[kk - 1] + frac * (z[kk] - z[kk - 1]), np.nan)
    return np.where(np.isnan(Tm[..., 0]), np.nan, d)


def all_products(T, depths, axis=1):
    return {"D26_m": isotherm_depth(T, depths, 26.0, axis), "D20_m": isotherm_depth(T, depths, 20.0, axis),
            "TCHP_kJcm2": tchp(T, depths, axis), "MLD_m": mld(T, depths, axis)}


def product_rmse(T_pred, T_true, depths):
    out = {}
    P, Q = all_products(T_pred, depths), all_products(T_true, depths)
    for k in P:
        m = ~np.isnan(P[k]) & ~np.isnan(Q[k])
        e = P[k][m] - Q[k][m]
        out[f"{k}_rmse"] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
        out[f"{k}_bias"] = float(np.mean(e)) if e.size else np.nan
    return out
