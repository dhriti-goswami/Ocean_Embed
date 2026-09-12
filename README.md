# OceanEmbed — Subsurface Temperature Reconstruction

CNN that reconstructs a subsurface temperature profile (15 depth levels) from
surface satellite observations (SST, SSS, SSH, wind-u, wind-v) over the Bay
of Bengal, validated independently against real ARGO float measurements.

Built for Smart India Hackathon 2026 (Problem Statement 26066).

## Status

🚧 Early setup — data inspection not yet complete. No trained model, no
results yet. This README will be updated as each stage is finished.

## Setup

```bash
git clone <repo-url>
cd oceanembed
pip install -r requirements.txt
```

## Repo layout

- `configs/` — hyperparameters and paths (edit these, not the code, to change a run)
- `data/` — local data cache (gitignored — see `data/README.md` for where to get it)
- `notebooks/` — exploration and inspection notebooks
- `src/data/` — loading, preprocessing, splitting
- `src/models/` — CNN architecture
- `src/eval/` — metrics and ARGO validation matching
- `results/` — saved figures and metrics
- `checkpoints/` — saved model weights (gitignored)

## Data

- Surface inputs: SST, SSS, SSH, wind (u, v) — Bay of Bengal region
- Target: temperature at 15 depth levels
- Validation: independent ARGO float profiles (never seen during training)

Exact resolution, time range, and source datasets to be documented here once
data inspection (Phase 1) is complete.
