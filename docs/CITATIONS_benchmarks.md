# Citations: prior work and benchmarks

> Reading warning: no number in this file is directly comparable with OceanEmbed's results.
> They come from different years, regions, inputs and validation sets. They are here to
> position the work, not to rank it.

## A. Satellite-based TCHP in the Indian Ocean

### A.1 Neural-network TCHP from satellite data (2012) — `VERIFIED-ABSTRACT`

Ali, M. M. et al. (2012). *A Neural Network Approach to Estimate Tropical Cyclone Heat
Potential in the Indian Ocean.* IEEE. https://ieeexplore.ieee.org/document/6189026/

From the abstract:

| fact | value |
|---|---|
| training data | more than 25,000 in-situ subsurface temperature profiles, 1997–2007 |
| inputs | satellite sea surface height anomaly, SST, climatological D26 |
| validation | more than 8,000 independent in-situ profiles, 2008–2009 |
| output | TCHP (a single value per location) |

Journal name, volume and pages: `NOT-CHECKED`.

### A.2 Satellite-altimetry TCHP validation, North Indian Ocean — `VERIFIED-PRIMARY` (excerpt)

https://www.aoml.noaa.gov/phod/docs/tchp-val.pdf

| fact | value |
|---|---|
| satellite vs in-situ TCHP, bias | 11.27 kJ cm⁻² |
| R² | 0.65 |
| RMSE | 20.95 kJ cm⁻² |
| scatter index (RMSE / mean) | 0.33 |

### A.3 TCHP atlas of the North Indian Ocean — `VERIFIED-ABSTRACT`

*An Atlas of the Tropical Cyclone Heat Potential of the North Indian Ocean.*
https://www.academia.edu/3145516/An_Atlas_of_the_Tropical_Cyclone_Heat_Potential_of_the_North_Indian_Ocean
— altimetry-based TCHP, 1993 onwards; reports daily RMSE ≈ 21 kJ/cm² and monthly ≈ 16.6 kJ/cm²
against in-situ (as summarized on the landing page).

### A.4 Operational use of D26 / TCHP — `VERIFIED-ABSTRACT`

Statistical–dynamical cyclone intensity schemes (SHIPS) use SST, D26/D20 and ocean heat
content as oceanic predictors, including for the North Indian Ocean.
https://www.sciencedirect.com/science/article/pii/S2590123025010849 ·
https://journals.ametsoc.org/view/journals/wefo/36/4/WAF-D-20-0104.1.xml

## B. Deep-learning subsurface reconstruction (SIH proposal reference list) — `NOT-CHECKED`

1. Meng et al. (2022). *Subsurface Temperature Reconstruction for the Global Ocean from 1993 to
   2020 Using Satellite Observations and Deep Learning.* Remote Sensing.
   https://doi.org/10.3390/rs14133198
2. *An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature
   Reconstruction.* https://arxiv.org/pdf/2605.00860
3. *Ocean temperature reconstruction from satellite observations in the North Atlantic using an
   explainable deep learning framework.* Int. J. Digital Earth.
   https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2632430
4. *Satellite-based reconstruction of high-resolution ocean subsurface temperature using
   spatiotemporal graph attention networks.*
   https://www.sciencedirect.com/science/article/pii/S1569843226001536
5. Zhang et al. (2023). *Deriving Sea Subsurface Temperature Fields From Satellite Remote Sensing
   Data Using a Generative Adversarial Network Model.* Earth and Space Science.
   https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002804
6. *Reconstructing high-resolution subsurface temperature of the global ocean using deep forest
   with combined remote sensing and in situ observations.*
   https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617

## C. Case-study event — `VERIFIED-PRIMARY`

India Meteorological Department press release, 1 Aug 2023: a well-marked low over the north
Bay of Bengal intensified into a depression and then a deep depression over the north-east Bay
of Bengal off the Bangladesh coast, centred near 21.2 °N, 91.2 °E at 0830 IST, expected to
cross the Bangladesh coast the same evening.
https://internal.imd.gov.in/press_release/20230801_pr_2460.pdf
