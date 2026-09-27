# Citations: methods

## Network components

| component | reference | tag |
|---|---|---|
| Fourier Neural Operator | Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR. | `NOT-CHECKED` |
| Vision Transformer | Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR. | `NOT-CHECKED` |
| U-Net | Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. | `NOT-CHECKED` |

## Uncertainty

| method | reference | tag |
|---|---|---|
| heteroscedastic (aleatoric) regression with a predicted variance | Kendall, A., Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS. | `NOT-CHECKED` |
| deep ensembles | Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. | `NOT-CHECKED` |

Implementation choices that are ours (no citation): detaching the mean inside the NLL, a
single per-seed rescaling factor fitted on validation days, and adding an ARGO-vs-GLORYS
representativeness term fitted on non-test days.

## Ocean products

| quantity | definition used | source | tag |
|---|---|---|---|
| TCHP | heat content of the water column above the 26 °C isotherm, $\rho c_p \int (T-26)\,dz$ | Leipper, D. F., Volgenau, D. (1972). *Hurricane heat potential of the Gulf of Mexico.* J. Phys. Oceanogr. The AOML validation document (see benchmarks) states this definition and attributes it to Leipper & Volgenau. | definition `VERIFIED-PRIMARY` via AOML document; bibliographic record `NOT-CHECKED` |
| D26, D20 | depth of the 26 °C / 20 °C isotherm | standard; D26 and D20 appear as oceanic predictors in operational cyclone models (see benchmarks) | `VERIFIED-ABSTRACT` (usage) |
| MLD | 0.5 °C temperature criterion relative to 10 m | common temperature-threshold convention; the threshold is a choice, not a universal standard | `NOT-CHECKED` |
| density | simplified nonlinear EOS (coefficients in NOTATION.md) | fitted by us for stability checks only; not TEOS-10 | – |
