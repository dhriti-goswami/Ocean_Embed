# Citations: datasets

| data | product | reference | tag |
|---|---|---|---|
| training target: subsurface temperature (and optional salinity) | GLORYS12 global ocean physics reanalysis (CMEMS) | https://doi.org/10.48670/moi-00021 | `FROM-PS` |
| SST | OSTIA, 0.05°, daily (CMEMS) | https://doi.org/10.48670/moi-00168 | `FROM-PS` |
| SSS | SMOS / SMAP multi-mission, 0.125°, daily (CMEMS) | https://doi.org/10.48670/moi-00051 | `FROM-PS` |
| SSH / SLA / ADT | DUACS, daily (CMEMS) | https://doi.org/10.48670/moi-00145 | `FROM-PS` |
| surface currents | OSCAR L4 v2.0, 0.25°, daily (PO.DAAC) | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0 | `FROM-PS` |
| 10 m winds | CCMP v3.1, 0.25°, 6-hourly (PO.DAAC) | https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 | `FROM-PS` |
| winds (not used as input, see METHODOLOGY §1) | ASCAT-C L2 coastal (PO.DAAC) | https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal | `FROM-PS` |
| in-situ validation | ARGO float profiles, Ifremer ERDDAP `ArgoFloats` | https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html | `VERIFIED-PRIMARY` (queried directly) |
| in-situ validation (PS alternative, not used) | Gridded ARGO, INCOIS Live Access Server | named in PS 26066 | `FROM-PS` |

ARGO data were collected and made freely available by the International Argo Program and the
national programs that contribute to it.

Regional box and period used: 5–22 °N, 80–100 °E, 1 Jun – 31 Aug 2023.
