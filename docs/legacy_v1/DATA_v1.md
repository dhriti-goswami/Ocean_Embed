# Data Sources

Confirmed via test pulls on 2026-09-12. All variables fetched for the
Bay of Bengal box: lon 80–95°E, lat 5–22°N.

## Sources

| Source | Variables | Native resolution | Access method |
|---|---|---|---|
| GLORYS12V1 (CMEMS) | temperature (35 native depths, 0.49–902m) | 0.083° (205×181 pts) | `copernicusmarine` Python API |
| GLORYS12V1 (CMEMS) | salinity, SSH, currents (u,v) — surface | 0.083° (205×181 pts) | `copernicusmarine` Python API |
| ERA5 (CDS) | 10m wind (u,v) | 0.25° (69×61 pts) | `cdsapi` Python API |
| GEBCO_2023 | bathymetry (elevation) | ~15 arc-sec (4080×3600 pts), static | GEBCO Grid Subsetting App (manual, emailed link) |

## Target depth levels (15)

0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

Interpolated from GLORYS's 35 native depth levels — not a 1:1 match, see
`docs/TRAINING_METHODOLOGY.md` (once written) for the interpolation approach.

## Status

- [x] Test pull confirmed for all 4 sources (1 day, small box)
- [x] 2-week pull confirmed (June 1–14, 2023) for pipeline development
- [ ] Full year (June 2023–?) pull — planned after 2-week pipeline validated
- [ ] Regridding to common resolution — not yet decided (0.083° vs 0.25°)

## Known issues to resolve

- Three different native grids (GLORYS 0.083°, ERA5 0.25°, GEBCO 15 arc-sec)
  require regridding onto one common grid before model input.
- ERA5 time coordinate is named `valid_time`, GLORYS uses `time` — needs
  harmonizing.
- GLORYS depth levels aren't round numbers (e.g. 0.494, 1.541m) — need
  interpolation to the 15 clean target depths.
