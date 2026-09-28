# Troubleshooting

## 1. Colab: `MessageError: credential propagation was unsuccessful`

**Cause:** the Google Drive permission popup was blocked, closed, or not all permissions were
granted.

**Fix:** re-run the cell, pick the same Google account you are signed into Colab with, click
**Select all** on the permissions screen. Allow pop-ups for colab.research.google.com. Drive is
optional in the notebook: if mounting fails it continues and you download the zip at the end.

## 2. Colab: data download fails / `gdown` error

**Cause:** the shared cube is not public.

**Fix:** in Google Drive, right-click the file → Share → General access → **Anyone with the
link** (Viewer). Or place the file in your own Drive and set `DATA_DRIVE` in section 2.

## 3. Colab: results disappeared

**Cause:** everything under `/content` is deleted when the runtime ends; writing directly to a
mounted Drive can also be lost if the session dies before syncing.

**Fix:** the notebook writes locally and copies to Drive after each stage (`backup()`), and the
last cell downloads one zip. Download it before closing the tab.

## 4. `No GPU` / training very slow

**Fix:** Runtime → Change runtime type → **T4 GPU**. On CPU a full run is ~10× slower; use
`configs/smoke.yaml` for a quick check.

## 5. `KeyError: Missing variables in cube`

**Cause:** the NetCDF lacks one of `analysed_sst, sos, sla, adt, u, v, uwnd, vwnd, thetao`, or
uses other names.

**Fix:** check with `xr.open_dataset(path)`. Coordinates `latitude/longitude` are accepted;
variable names must match [`data/README.md`](data/README.md).

## 6. ARGO download fails (`ARGO fetch failed`)

**Cause:** Ifremer ERDDAP is unreachable or slow, or the QC-filtered query is rejected.

**Fix:** the code retries without QC columns automatically. If the server is down, re-run later;
once fetched, the data is cached in `results/argo_cache.csv` and reused.

## 7. GitHub rejects the data file

**Cause:** GitHub refuses files over 100 MB; the float64 cube is ~185 MB.

**Fix:** don't commit data (it is git-ignored). Share it via Drive, or compress to float32 +
zlib (typically well under 100 MB):
```python
enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf("cube_f32.nc", encoding=enc)
```

## 8. GitHub token: `403` / `404` on a repository you can open in the browser

**Cause:** fine-grained personal access tokens only reach repositories owned by the token's
account; being a collaborator on someone else's personal repository is not enough.

**Fix:** use a classic token with the `repo` scope, or have the owner create the token. Revoke
tokens when done.

## 9. `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Cause:** inference called outside `torch.no_grad()` (defect #8 in
[`docs/TRAINING_METHODOLOGY.md`](docs/TRAINING_METHODOLOGY.md#8-defects-found-and-fixed)).

**Fix:** use `oceanembed.inference.predict_all`, which is decorated with `@torch.no_grad()`.
`make test-all` catches regressions.

## 10. Tests

```bash
make test        # fast unit tests
make test-all    # + end-to-end on a synthetic cube (~2 min)
make smoke       # quick pipeline check without pytest
```

## 11. Backend shows old data / `FileNotFoundError: data/ocean_temperature.zarr`

**Cause:** the backend reads a Zarr store, not the model's NetCDF.

**Fix:** run `python -m oceanembed export-backend --nc <nc> --out <DATA_PATH>` with the path
from `backend/app/config.py` (relative to the directory the backend is started from), then
restart the backend (it caches the dataset in memory).
