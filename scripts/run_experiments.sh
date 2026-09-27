#!/usr/bin/env bash
# Full experiment: 4 models x 3 seeds, evaluation, ARGO, uncertainty, case study.
# Usage: bash scripts/run_experiments.sh data/processed/cube.nc   (GPU recommended, ~20-25 min)
set -euo pipefail
DATA=${1:-data/processed/cube.nc}
RUNS=artifacts/runs; RES=results; OUT=artifacts/outputs
# run name : config  (names match docs/RESULTS.md)
for pair in unet:ablation_unet oceanembed_nophys:ablation_nophys \
            oceanembed_constraints:ablation_constraints oceanembed_cyclone:main; do
  name=${pair%%:*}; cfg=${pair#*:}
  for s in 0 1 2; do
    python -W ignore -m oceanembed train --config configs/$cfg.yaml --data "$DATA" --out $RUNS/${name}_s$s --seed $s
  done
done
python -W ignore -m oceanembed evaluate    --data "$DATA" --runs $RUNS/* --out $RES
python -W ignore -m oceanembed argo        --data "$DATA" --runs $RUNS/* --out $RES --cache $RES/argo_cache.csv
python -W ignore -m oceanembed uncertainty --data "$DATA" --runs $RUNS/oceanembed_cyclone_s* --out $RES --outputs $OUT --argo_cache $RES/argo_cache.csv
python -W ignore -m oceanembed casestudy   --data "$DATA" --runs $RUNS/oceanembed_cyclone_s* --out $RES --argo_cache $RES/argo_cache.csv
echo "done: results in $RES, NetCDF in $OUT"
