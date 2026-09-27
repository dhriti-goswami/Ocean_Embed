# Citations

## Datasets (as specified in PS 26066)

| data | product | reference |
|---|---|---|
| Training target: subsurface temperature | GLORYS12 global ocean reanalysis (CMEMS) | https://doi.org/10.48670/moi-00021 |
| SST | OSTIA (CMEMS) | https://doi.org/10.48670/moi-00168 |
| SSS | SMOS / SMAP multi-mission (CMEMS) | https://doi.org/10.48670/moi-00051 |
| SSH / SLA / ADT | DUACS (CMEMS) | https://doi.org/10.48670/moi-00145 |
| Surface currents | OSCAR L4 v2.0 (PO.DAAC) | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0 |
| 10 m winds | CCMP v3.1 (PO.DAAC) | https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 |
| Independent validation | ARGO float profiles via Ifremer ERDDAP | https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html |

ARGO data were collected and made freely available by the International Argo Program and the
national programs that contribute to it.

## Methods used

- Fourier Neural Operator — Li, Z. et al. (2021). *Fourier Neural Operator for Parametric
  Partial Differential Equations.* ICLR.
- Vision Transformer — Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale.* ICLR.
- U-Net — Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for
  Biomedical Image Segmentation.* MICCAI.
- Heteroscedastic uncertainty — Kendall, A., Gal, Y. (2017). *What Uncertainties Do We Need in
  Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Deep ensembles — Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
- Tropical cyclone heat potential definition — Leipper, D. F., Volgenau, D. (1972).
  *Hurricane heat potential of the Gulf of Mexico.* J. Phys. Oceanogr.

## Prior work on satellite-based TCHP in the Indian Ocean

- Ali, M. M. et al. (2012). *A Neural Network Approach to Estimate Tropical Cyclone Heat
  Potential in the Indian Ocean.* IEEE. https://ieeexplore.ieee.org/document/6189026/ —
  neural network from satellite SSH anomaly, SST and climatological D26; validated against
  independent in-situ profiles.
- Satellite-altimetry TCHP validation for the North Indian Ocean (RMSE ≈ 21 kJ/cm² vs in-situ):
  https://www.aoml.noaa.gov/phod/docs/tchp-val.pdf
- *An Atlas of the Tropical Cyclone Heat Potential of the North Indian Ocean.*
  https://www.academia.edu/3145516/An_Atlas_of_the_Tropical_Cyclone_Heat_Potential_of_the_North_Indian_Ocean

These methods estimate TCHP as a 2D field. OceanEmbed reconstructs the full temperature column
and trains the TCHP/D26/D20/MLD derived from it; numbers are not directly comparable across
studies (different periods, inputs and validation sets).

## Subsurface reconstruction references from the SIH proposal

1. Meng et al. (2022). *Subsurface Temperature Reconstruction for the Global Ocean from 1993 to
   2020 Using Satellite Observations and Deep Learning.* Remote Sensing.
   https://doi.org/10.3390/rs14133198
2. *An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature
   Reconstruction.* https://arxiv.org/pdf/2605.00860
3. *Ocean temperature reconstruction from satellite observations in the North Atlantic using an
   explainable deep learning framework.* Int. J. Digital Earth.
   https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2632430
4. *Satellite-based reconstruction of high-resolution ocean subsurface temperature using
   spatiotemporal graph attention networks.* https://www.sciencedirect.com/science/article/pii/S1569843226001536
5. Zhang et al. (2023). *Deriving Sea Subsurface Temperature Fields From Satellite Remote Sensing
   Data Using a Generative Adversarial Network Model.* Earth and Space Science.
   https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002804
6. *Reconstructing high-resolution subsurface temperature of the global ocean using deep forest
   with combined remote sensing and in situ observations.*
   https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617

## Case-study event

India Meteorological Department press release, 1 Aug 2023: depression intensifying into a deep
depression over the north-east Bay of Bengal off the Bangladesh coast, near 21.2 °N, 91.2 °E.
https://internal.imd.gov.in/press_release/20230801_pr_2460.pdf
