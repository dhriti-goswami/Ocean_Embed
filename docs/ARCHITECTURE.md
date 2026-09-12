# Architecture

**Status: not yet finalized.** This will be filled in once the model
architecture is actually implemented and tested (Phase 2+ of the project).

Planned shape (subject to change as we build):

- **Input**: stacked 2D surface grids — SST, SSS, SSH, wind-u, wind-v — over
  the Bay of Bengal region, one "channel" per variable (same idea as R/G/B
  channels in an image).
- **Model**: CNN encoder that compresses the surface grid into a latent
  representation per grid cell, decoded into a temperature prediction at
  15 depth levels.
- **Output**: predicted temperature profile (15 values) per grid cell, per day.

No diagram yet — will be added once the actual layer structure is decided
and confirmed working, not before.
