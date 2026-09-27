# Notation and abbreviations

Index of every symbol, abbreviation and variable name used in
[`METHODOLOGY.md`](METHODOLOGY.md), [`RESULTS.md`](RESULTS.md) and the `oceanembed` package.
Each entry names the code identifier where one exists, so a reader can go from a symbol to the
line that computes it. Tables follow the order in which quantities are introduced.

---

## 1. Grid and indices

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $t$ | day index (1 Jun 2023 = 0) | day | `data.times` |
| $z_k$ | depth of GLORYS level $k$, $k = 1..35$ (0.494–902.3 m) | m | `data.depths` |
| $\Delta z_k$ | thickness of layer $k$ (edges halfway between levels, top edge 0) | m | `layer_thickness()` |
| $(i, j)$ | grid cell, 0.25°; $69 \times 81$, padded to $72 \times 88$ | – | `data.H, data.W, data.Hp, data.Wp` |
| $M_{kij}$ | 3D ocean mask: level $k$ valid at cell $(i,j)$ on every day | bool | `mask3d` |
| $M_{ij}$ | 2D surface mask (ocean + inputs valid ≥ 90 % of days) | bool | `mask2d` |

## 2. Inputs

| Symbol | Meaning | Unit | Code (cube variable) |
|---|---|---|---|
| SST | sea surface temperature (OSTIA) | °C (K in file) | `analysed_sst` |
| SSS | sea surface salinity (SMOS/SMAP) | PSU | `sos` |
| $\eta'$ | sea level anomaly (DUACS) | m | `sla` |
| ADT | absolute dynamic topography (DUACS) | m | `adt` |
| $u_s, v_s$ | surface currents (OSCAR) | m s⁻¹ | `u`, `v` |
| $u_{10}, v_{10}$ | 10 m winds (CCMP, daily mean) | m s⁻¹ | `uwnd`, `vwnd` |
| $\mathbf{x}_t$ | model input: 8 fields at $t-2, t-1, t$ + sin/cos day-of-year + lat + lon = 28 channels | normalized | `data.X`, `channel_names` |
| $h$ | number of previous days stacked (2) | – | `history` / `history_days` |

## 3. Targets and predictions

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $T_{k}$ | potential temperature at level $k$ (GLORYS target) | °C | `thetao`, `data.T` |
| $S_{k}$ | salinity at level $k$ (optional target) | PSU | `so`, `data.S` |
| $\hat T_k$ | predicted temperature | °C | `temperature` (output NetCDF) |
| $\hat\sigma_k$ | predicted 1-σ temperature uncertainty | °C | `temperature_std` |
| $\mu_d, s_d$ | train-period mean and std of $T$ at level $d$ (normalization) | °C | `t_mean`, `t_std` |
| $\mathbf{z}$ | latent ocean embedding, $128 \times 9 \times 11$ | – | `model.encode()`, `oceanembed_embeddings.nc` |

## 4. Ocean products

| Symbol | Meaning | Definition | Unit | Code |
|---|---|---|---|---|
| D26 | depth of the 26 °C isotherm | first downward crossing of 26 °C, linear interpolation | m | `isotherm_depth(T, z, 26)` |
| D20 | depth of the 20 °C isotherm (thermocline proxy) | as above for 20 °C | m | `isotherm_depth(T, z, 20)` |
| TCHP | tropical cyclone heat potential | $\rho_0 c_p \sum_k \Delta z_k \,(T_k - 26)^+ / 10^7$ | kJ cm⁻² | `tchp()` |
| MLD | mixed layer depth | depth where $T$ first drops $\Delta T = 0.5$ °C below $T_{10}$ | m | `mld()`, `compute_mld()` |
| $T_{10}$ | reference temperature at the level nearest 10 m | – | °C | `k10` |
| $\rho$ | density, simplified nonlinear EOS: $\rho_0[1 - 1.7\times10^{-4}(T-10) - 4.3\times10^{-6}(T-10)^2 + 7.6\times10^{-4}(S-35)]$ | – | kg m⁻³ | `density()` |
| $\rho_0$ | reference density | 1025 | kg m⁻³ | `RHO0` |
| $c_p$ | specific heat of seawater | 3990 | J kg⁻¹ K⁻¹ | `CP` |
| $h_s$ | steric height, $-\sum_k \Delta z_k (\rho_k - \bar\rho_k)/\rho_0$ over full 0–902 m columns | – | m | `steric_height()` |
| $\bar\rho_k$ | train-period mean density profile | – | kg m⁻³ | `reference_density_profile()` |

## 5. Loss terms

| Symbol | Meaning | Code / config key |
|---|---|---|
| $\mathcal{L}_{\text{data}}$ | masked MSE on normalized $T$ (+ 0.5 × salinity) | `masked_mse()` |
| $\mathcal{L}_{\text{NLL}}$ | Gaussian NLL of the uncertainty head, mean detached; weight 0.1 | `gaussian_nll()`, `uncertainty` |
| $\tilde Q$ | soft (differentiable) version of product $Q$ using $\sigma((T-\theta)/\tau)$ or softplus | `soft_quantities()` |
| $\tau$ | softness of the soft products (0.3 °C; 0.1 °C for MLD) | `tau` |
| $\mathcal{L}_{Q}$ | $\frac{1}{4}\sum_Q ((\tilde Q_{\text{pred}} - \tilde Q_{\text{true}})/s_Q)^2$, $s_Q$ = 10 m, 15 m, 5 m, 10 kJ cm⁻² | `quantity_terms()`, `w_quant` |
| $\mathcal{L}_{\text{stab}}$ | squared density inversions > 0.01 kg m⁻³ (or $T$ inversions > 0.05 °C below 150 m without salinity) | `stability_term()`, `w_stab` |
| $\mathcal{L}_{\text{ster}}$ | $\max(0, r_{\text{true}} - r_{\text{pred}})$, $r$ = spatial correlation of $h_s$ with $\eta'$ | `steric_term()`, `w_steric` |
| $\mathcal{L}_{\text{MLD}}$ | v2 constraint: $(T_k - T_{10})^2$ inside the target mixed layer | `mld_term()`, `w_mld` |
| $r(e)$ | physics ramp: 0 for $e \le$ `warmup`, then linear to 1 over `ramp` epochs | `ramp` |

Total: $\mathcal{L} = \mathcal{L}_{\text{data}} + 0.1\,\mathcal{L}_{\text{NLL}} + r(e)\,[w_{\text{stab}}\mathcal{L}_{\text{stab}} + w_{\text{ster}}\mathcal{L}_{\text{ster}} + w_Q \mathcal{L}_Q]$ (cyclone mode).

## 6. Uncertainty

| Symbol | Meaning | Code |
|---|---|---|
| $\hat\sigma_i$ | std predicted by seed $i$ | `predict_all(..., return_sigma=True)` |
| $k_i$ | calibration factor for seed $i$: RMS of $(T-\hat T_i)/\hat\sigma_i$ on validation days | `calibration_factor_k` |
| $\hat\sigma_{\text{ens}}^2$ | $\overline{k_i^2\hat\sigma_i^2} + \mathrm{Var}_i(\hat T_i)$ | `ensemble()` |
| $\sigma_{\text{repr}}$ | ARGO − GLORYS RMS per depth band (non-test days) | `uncertainty_representativeness.json` |
| coverage$_p$ | share of true values with $|e| \le \Phi^{-1}(0.5 + p/2)\,\hat\sigma$ | `coverage()` |

## 7. Evaluation

| Term | Meaning |
|---|---|
| climatology | train-period mean of $T$ at each cell and level; the no-skill baseline |
| test days | 16–31 Aug 2023; never used for training or checkpoint selection |
| deep_T_inv_pct | % of adjacent level pairs below 150 m where $T$ increases > 0.05 °C with depth |
| density_inv_pct | % of adjacent level pairs where $\rho$ decreases > 0.01 kg m⁻³ with depth (S = 35 if no salinity) |
| all_days / test_days_only | ARGO subsets: all 92 days (includes training days) / test days only |

---

## Abbreviations

| Abbreviation | Expansion |
|---|---|
| ADT | Absolute Dynamic Topography |
| ARGO | global array of profiling floats (not an acronym) |
| BoB | Bay of Bengal |
| CCMP | Cross-Calibrated Multi-Platform (winds) |
| CF | Climate and Forecast (NetCDF conventions) |
| CMEMS | Copernicus Marine Environment Monitoring Service |
| CNN | Convolutional Neural Network |
| D20 / D26 | depth of the 20 °C / 26 °C isotherm |
| DUACS | Data Unification and Altimeter Combination System |
| EOS | Equation of State (of seawater) |
| ERDDAP | Environmental Research Division's Data Access Program (data server) |
| FNO | Fourier Neural Operator |
| GLORYS | Global Ocean Reanalysis and Simulation (CMEMS) |
| IMD | India Meteorological Department |
| INCOIS | Indian National Centre for Ocean Information Services |
| MAE / RMSE | Mean Absolute Error / Root Mean Square Error |
| MLD | Mixed Layer Depth |
| NLL | Negative Log-Likelihood |
| OSCAR | Ocean Surface Current Analysis Real-time |
| OSTIA | Operational Sea Surface Temperature and Ice Analysis |
| PINN | Physics-Informed Neural Network |
| PO.DAAC | Physical Oceanography Distributed Active Archive Center (NASA) |
| QC | Quality Control |
| SLA / SSH | Sea Level Anomaly / Sea Surface Height |
| SMOS / SMAP | Soil Moisture and Ocean Salinity / Soil Moisture Active Passive (satellites) |
| SSS / SST | Sea Surface Salinity / Temperature |
| TCHP | Tropical Cyclone Heat Potential |
| ViT | Vision Transformer |
