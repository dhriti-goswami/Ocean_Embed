# Citations index

Every external fact used by this project traces to one of the files below. Each entry
carries a verification tag:

| tag | meaning |
|---|---|
| `VERIFIED-PRIMARY` | read in the original source (publisher page, PDF or official release) during this project |
| `VERIFIED-ABSTRACT` | read in the source's abstract / publisher landing page only |
| `FROM-PS` | taken as given in the problem statement (PS 26066); not independently re-resolved |
| `NOT-CHECKED` | bibliographic record written from memory or taken from the proposal's reference list; **check before quoting in a paper** |

| file | covers |
|---|---|
| [`CITATIONS_data.md`](CITATIONS_data.md) | input, target and validation datasets |
| [`CITATIONS_methods.md`](CITATIONS_methods.md) | network components, uncertainty methods, ocean-product definitions |
| [`CITATIONS_benchmarks.md`](CITATIONS_benchmarks.md) | prior satellite TCHP and subsurface-reconstruction work, and the case-study event |

## Findings that changed the design

1. **Satellite-based TCHP estimation for the Indian Ocean already exists**, including a neural
   network from SSH anomaly, SST and climatological D26 (2012). Consequence: "estimating TCHP
   from satellites" is not the contribution. The contribution is framed as reconstructing the
   full temperature column, training the products derived from it, and attaching a calibrated
   uncertainty.
2. **A published satellite-altimetry TCHP RMSE (≈ 21 kJ/cm² vs in-situ, North Indian Ocean)
   is not comparable** with our 13–16.5 kJ/cm² (different years, inputs, validation sets).
   Consequence: no "we beat X" claim anywhere in the docs.
3. **A deep depression formed inside the data window** (IMD, 1 Aug 2023, NE Bay of Bengal).
   Consequence: the real-event case study.
4. **GLORYS assimilates ARGO.** Consequence: ARGO validation is described as independent of the
   model's inputs and training, not of its target, and test-day ARGO numbers are reported
   separately.
