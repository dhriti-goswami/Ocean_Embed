"""End-to-end smoke test on a synthetic cube with the real file's shapes (CPU, ~2 min).

    python tests/smoke_test.py
"""
import os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic_cube import make

tmp = tempfile.mkdtemp()
cube = make(os.path.join(tmp, "cube.nc"), with_salinity="--salinity" in sys.argv)
runs = [os.path.join(tmp, "runs", n) for n in ("unet", "oceanembed_pinn")]
steps = [
    ["-m", "oceanembed.train", "--data", cube, "--out", runs[0], "--variant", "unet", "--epochs", "2"],
    ["-m", "oceanembed.train", "--data", cube, "--out", runs[1], "--variant", "oceanembed",
     "--physics", "--epochs", "3", "--warmup", "1"],
    ["-m", "oceanembed.evaluate", "--data", cube, "--runs", *runs, "--out", os.path.join(tmp, "results")],
    ["-m", "oceanembed.infer", "--data", cube, "--ckpt", os.path.join(runs[1], "best.pt"),
     "--out", os.path.join(tmp, "outputs")],
]
for s in steps:
    print(">>", " ".join(s[:2]))
    subprocess.run([sys.executable, "-W", "ignore", *s], check=True)
print("\nSMOKE TEST PASSED ->", tmp)
