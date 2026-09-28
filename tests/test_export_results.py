import json

import numpy as np
import pandas as pd

from oceanembed.export_results import export


def test_seeds_aggregated_and_json_is_strict(tmp_path):
    pd.DataFrame({"model": ["climatology", "oceanembed_cyclone_s0", "oceanembed_cyclone_s1"],
                  "rmse": [0.97, 0.62, 0.64], "bias": [0.3, -0.1, np.nan]}
                 ).to_csv(tmp_path / "glorys_test_overall.csv", index=False)
    path, data = export(str(tmp_path), str(tmp_path / "validation.json"))
    main = [r for r in data["glorys_test"] if r["model_type"] == "oceanembed_cyclone"][0]
    assert main["n_seeds"] == 2 and abs(main["rmse"]["mean"] - 0.63) < 1e-9
    json.loads(open(path).read())                                # no NaN tokens
    assert "argo" not in data                                    # missing tables are skipped
