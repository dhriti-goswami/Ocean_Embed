"""Physics-informed constraints (the "PINN" block of the architecture).

All terms act on de-normalized predictions (deg C, PSU) and are soft penalties added to
the data loss. Implemented:

1. Mixed-layer-depth consistency - inside the mixed layer (depth from the target profile,
   0.5 deg C criterion vs. 10 m) the predicted column must be vertically well mixed.
2. Thermodynamic (static) stability - density must not decrease with depth. Needs
   salinity; without it, falls back to "temperature must not increase with depth below
   150 m" (shallower inversions are real in the Bay of Bengal's fresh upper layer).
3. Steric-height / geostrophic consistency - the steric height implied by the predicted
   density column should co-vary spatially with the observed sea level anomaly (SSH sets
   the surface geostrophic flow). Penalized only when the prediction is *less* consistent
   with the observed SSH than the reanalysis target itself is.

Mass conservation needs a predicted velocity field, which this model does not produce;
it is left for future work rather than approximated.

mode="cyclone" (OceanEmbed v2.1) replaces the mixed-layer homogeneity term with
*cyclone-aware ocean-state losses*: differentiable (soft) versions of the quantities
cyclone and ocean forecasters use - 26 degC isotherm depth (D26), Tropical Cyclone Heat
Potential (TCHP), 20 degC isotherm depth (D20, thermocline) and mixed layer depth - matched
to the same quantities computed from the target. Unlike inequality constraints (which a
stable-but-wrong profile satisfies), these pull the profile toward the correct structure.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RHO0 = 1025.0
CP = 3990.0
# scales used to make the quantity errors comparable (typical acceptable errors)
QUANT_SCALES = {"D26": 10.0, "D20": 15.0, "MLD": 5.0, "TCHP": 10.0}


def density(T, S=None):
    """Simplified nonlinear equation of state (kg m^-3). Adequate for stability checks."""
    if S is None:
        S = torch.full_like(T, 35.0)
    dT = T - 10.0
    return RHO0 * (1.0 - 1.7e-4 * dT - 4.3e-6 * dT ** 2 + 7.6e-4 * (S - 35.0))


def layer_thickness(depths):
    d = np.asarray(depths, dtype=np.float64)
    edges = np.concatenate([[0.0], 0.5 * (d[1:] + d[:-1]), [d[-1] + 0.5 * (d[-1] - d[-2])]])
    return np.diff(edges).astype(np.float32)


def masked_corr(a, b, m):
    w = m.float()
    n = w.sum((1, 2)).clamp_min(1.0)
    am = (a * w).sum((1, 2)) / n
    bm = (b * w).sum((1, 2)) / n
    ac = (a - am[:, None, None]) * w
    bc = (b - bm[:, None, None]) * w
    cov = (ac * bc).sum((1, 2))
    return cov / ((ac.pow(2).sum((1, 2)).sqrt() * bc.pow(2).sum((1, 2)).sqrt()) + 1e-8)


class PhysicsLoss(nn.Module):
    def __init__(self, depths, norm, rho_ref, has_salinity, sla_channel,
                 w_mld=0.05, w_stab=1.0, w_steric=0.1, stab_tol=0.01,
                 mode="cyclone", w_quant=0.1, tau=0.3):
        super().__init__()
        self.mode, self.w_quant, self.tau = mode, w_quant, tau
        self.D = len(depths)
        self.has_s = has_salinity
        self.sla_channel = sla_channel
        self.w = dict(mld=w_mld, stability=w_stab, steric=w_steric)
        self.stab_tol = stab_tol
        reg = lambda n, a: self.register_buffer(n, torch.as_tensor(np.asarray(a), dtype=torch.float32))
        reg("depths", depths)
        reg("dz", layer_thickness(depths))
        reg("rho_ref", rho_ref)
        reg("t_mean", norm["t_mean"]); reg("t_std", norm["t_std"])
        if has_salinity:
            reg("s_mean", norm["s_mean"]); reg("s_std", norm["s_std"])
        self.k10 = int(np.argmin(np.abs(np.asarray(depths) - 10.0)))

    def denorm(self, out):
        D = self.D
        T = out[:, :D] * self.t_std[None, :, None, None] + self.t_mean[None, :, None, None]
        S = (out[:, D:2 * D] * self.s_std[None, :, None, None] + self.s_mean[None, :, None, None]
             if self.has_s else None)
        return T, S

    # --- cyclone-aware soft ocean-state quantities --------------------------
    def soft_quantities(self, T, m3):
        w = m3.float() * self.dz[None, :, None, None]
        tau = self.tau
        q = {
            "D26": (torch.sigmoid((T - 26.0) / tau) * w).sum(1),
            "D20": (torch.sigmoid((T - 20.0) / tau) * w).sum(1),
            "TCHP": RHO0 * CP * (F.softplus(T - 26.0, beta=4.0) * w).sum(1) / 1e7,   # kJ/cm^2
            "MLD": (torch.sigmoid((T - (T[:, self.k10:self.k10 + 1] - 0.5)) / 0.1) * w).sum(1),
        }
        return q

    def quantity_terms(self, T, T_true, m3):
        surf = m3[:, 0].float()
        qp = self.soft_quantities(T, m3)
        with torch.no_grad():
            qt = self.soft_quantities(T_true, m3)
        n = surf.sum().clamp_min(1)
        return {f"q_{k}": ((((qp[k] - qt[k]) / QUANT_SCALES[k]) ** 2) * surf).sum() / n for k in qp}

    # --- individual terms -------------------------------------------------
    def mld_term(self, T, m3, mld):
        in_ml = (self.depths[None, :, None, None] < mld[:, None]) & m3
        diff = T - T[:, self.k10:self.k10 + 1]
        return (diff.pow(2) * in_ml).sum() / in_ml.sum().clamp_min(1)

    def stability_term(self, T, S, m3):
        pair = m3[:, :-1] & m3[:, 1:]
        if S is not None:
            rho = density(T, S)
            viol = F.relu(rho[:, :-1] - rho[:, 1:] - self.stab_tol)       # upper denser than lower
        else:
            deep = (self.depths[1:] >= 150.0)[None, :, None, None]
            pair = pair & deep
            viol = F.relu(T[:, 1:] - T[:, :-1] - 0.05)                     # warming with depth
        return (viol.pow(2) * pair).sum() / pair.sum().clamp_min(1)

    def steric_height(self, T, S, m3):
        rho = density(T, S)
        anom = (rho - self.rho_ref[None, :, None, None]) / RHO0
        return -(anom * self.dz[None, :, None, None] * m3).sum(1)            # metres

    def steric_term(self, T, S, T_true, S_true, m3, sla):
        full = m3.all(1)                                                     # full 0-900 m column
        if full.sum() < 10:
            return T.new_zeros(())
        c_pred = masked_corr(self.steric_height(T, S, m3), sla, full)
        with torch.no_grad():
            c_true = masked_corr(self.steric_height(T_true, S_true, m3), sla, full)
        return F.relu(c_true - c_pred).mean()

    # ----------------------------------------------------------------------
    def forward(self, out, y, m3, mld, x):
        T, S = self.denorm(out)
        T_true, S_true = self.denorm(y)
        terms = {"stability": self.stability_term(T, S, m3)}
        if self.mode == "constraints":
            terms["mld"] = self.mld_term(T, m3, mld)
        if self.sla_channel is not None:
            terms["steric"] = self.steric_term(T, S, T_true, S_true, m3, x[:, self.sla_channel])
        total = sum(self.w[k] * v for k, v in terms.items())
        if self.mode == "cyclone":
            q = self.quantity_terms(T, T_true, m3)
            total = total + self.w_quant * sum(q.values()) / len(q)
            terms.update(q)
        return total, {k: float(v.detach()) for k, v in terms.items()}


def reference_density_profile(data):
    """Mean training-period density per depth level (for steric-height anomalies)."""
    tr = data.idx["train"]
    T = torch.from_numpy(np.nan_to_num(data.T[tr], nan=0.0))
    S = torch.from_numpy(np.nan_to_num(data.S[tr], nan=35.0)) if data.has_salinity else None
    rho = density(T, S).numpy()
    m = data.mask3d[None].repeat(len(tr), 0)
    return np.array([rho[:, d][m[:, d]].mean() if m[:, d].any() else RHO0
                     for d in range(len(data.depths))], np.float32)
