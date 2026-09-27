"""Quick end-to-end check on a synthetic cube (CPU, ~2 min). Same as `make smoke`.

    python scripts/smoke_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oceanembed.cli import main  # noqa: E402
from oceanembed.data.synthetic import make  # noqa: E402

tmp = tempfile.mkdtemp()
cube = make(os.path.join(tmp, "cube.nc"))
runs = [os.path.join(tmp, "runs", n) for n in ("unet_s0", "main_s0")]
steps = [
    ["train", "--config", "configs/ablation_unet.yaml", "--data", cube, "--out", runs[0], "--epochs", "2"],
    ["train", "--config", "configs/smoke.yaml", "--data", cube, "--out", runs[1]],
    ["evaluate", "--data", cube, "--runs", *runs, "--out", os.path.join(tmp, "results")],
    ["infer", "--data", cube, "--ckpt", os.path.join(runs[1], "best.pt"), "--out", os.path.join(tmp, "outputs")],
]
for s in steps:
    print(">>", s[0])
    assert main(s) == 0
print("\nSMOKE TEST PASSED ->", tmp)
