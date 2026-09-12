# OceanEmbed — Subsurface Temperature Reconstruction

CNN that reconstructs a subsurface temperature profile (15 depth levels) from
surface satellite observations over the Bay of Bengal, independently
validated against real ARGO float measurements.

Built for Smart India Hackathon 2026 (Problem Statement 26066, Team BitShifters98).

## Status

🚧 **Early stage.** Data inspection in progress. No trained model, no
results yet. Sections below are placeholders until each stage is actually
done — nothing here is invented ahead of the real work.

## Key capabilities (target, not yet achieved)

- Predict temperature at 15 depth levels from 5 surface variables (SST, SSS, SSH, wind-u, wind-v)
- Independent validation against held-out ARGO float profiles
- Depth-wise error breakdown (RMSE, MAE, R², correlation, bias) — not just an overall average

## Model performance

Not available yet — no model has been trained. This table will be filled in
with real, measured validation numbers once training and ARGO validation
are complete.

| Metric | Value |
|---|---|
| RMSE | TBD |
| MAE | TBD |
| R² | TBD |
| Bias | TBD |

## Quick start

```bash
git clone https://github.com/dhriti-goswami/Ocean_Embed.git
cd Ocean_Embed
pip install -r requirements.txt
```

Note: this only installs dependencies. There is no runnable training script
yet — that comes after data inspection and preprocessing are done.

## Project structure

```
oceanembed/
├── configs/          # hyperparameters & paths (edit these, not the code, to change a run)
├── data/             # local data cache — gitignored, see data/README.md
├── docs/             # architecture & training methodology write-ups (filled in as we build)
├── notebooks/        # exploration & inspection notebooks
├── src/
│   ├── data/         # loading, preprocessing, train/val/test splitting
│   ├── models/       # CNN architecture
│   └── eval/         # metrics + ARGO validation matching
├── results/
│   ├── figures/      # loss curves, depth-wise error plots, profile comparisons
│   └── metrics/      # saved numeric results (json/csv)
└── checkpoints/      # saved model weights — gitignored
```

## Data

- Surface inputs: SST, SSS, SSH, wind (u, v) — Bay of Bengal region
- Target: temperature at 15 depth levels
- Validation: independent ARGO float profiles, never seen during training

Exact resolution, time range, and source datasets will be documented here
once data inspection (Phase 1) is complete — see `data/README.md`.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — model design (in progress)
- [`docs/TRAINING_METHODOLOGY.md`](docs/TRAINING_METHODOLOGY.md) — training approach (in progress)

## Requirements

See `requirements.txt`. Python 3.10+ recommended.

## Disclaimer

Research/hackathon prototype. Not for operational or safety-critical use.
