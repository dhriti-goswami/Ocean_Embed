"""Single entry point:  python -m oceanembed <command> [options]

  train        train one model            (--config configs/main.yaml --data cube.nc --out runs/x)
  evaluate     held-out days vs GLORYS    (temperature per depth + TCHP/D26/D20/MLD)
  argo         independent ARGO floats    (profiles + products)
  uncertainty  calibrate + ensemble seeds, write final NetCDF (temperature + temperature_std)
  casestudy    1 Aug 2023 deep depression, NE Bay of Bengal
  infer        single-model NetCDF + embeddings
  export-backend  convert the NetCDF product into the Zarr store the FastAPI backend reads
  synthetic    write a synthetic cube with the real file's shapes (for tests)
"""
import sys


def _synthetic(argv):
    from oceanembed.data.synthetic import make
    out = argv[0] if argv and not argv[0].startswith("-") else "synthetic_cube.nc"
    print("wrote", make(out, with_salinity="--salinity" in argv))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        from oceanembed.train.trainer import parse, train
        train(parse(rest))
    elif cmd == "evaluate":
        from oceanembed.eval.glorys import main as m; m(rest)
    elif cmd == "argo":
        from oceanembed.eval.argo import main as m; m(rest)
    elif cmd == "uncertainty":
        from oceanembed.eval.uncertainty import main as m; m(rest)
    elif cmd == "casestudy":
        from oceanembed.eval.casestudy import main as m; m(rest)
    elif cmd == "infer":
        from oceanembed.inference import main as m; m(rest)
    elif cmd == "export-backend":
        from oceanembed.export_backend import main as m; m(rest)
    elif cmd == "synthetic":
        _synthetic(rest)
    else:
        print(f"unknown command '{cmd}'\n{__doc__}"); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
