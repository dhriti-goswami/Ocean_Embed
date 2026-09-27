"""OceanEmbed: satellite-embedding reconstruction of subsurface ocean temperature.

Subpackages
  data      harmonized data cube loading, masks, splits, synthetic test cube
  models    ViT + FNO encoder, U-Net decoder, uncertainty head
  physics   physics-informed losses and ocean products (TCHP, D26, D20, MLD)
  train     training loop
  eval      GLORYS metrics, ARGO validation, uncertainty calibration, case study, figures
"""
__version__ = "2.1.0"
