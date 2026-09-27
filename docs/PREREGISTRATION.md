# Pre-registration: multi-year experiment (not yet run)

Written **before** the multi-year experiment is run and before any of its test numbers are
seen. Its purpose is to remove the discretion that turns several runs into one cherry-picked
result. Any change after test metrics have been seen must be added as an amendment at the
bottom, with the reason.

> **The v2.1 results in [`RESULTS.md`](RESULTS.md) were not pre-registered.** They come from an
> iterative development process (see the defect log in
> [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md#8-defects-found-and-fixed)). The test
> window (16–31 Aug 2023) was looked at more than once during development, so those numbers
> should be read as a proof of concept, not as a confirmatory result. This document applies
> from the next experiment on.

---

## 1. Why this experiment

v2.1 was trained on 61 days with almost no storms. In the 1 Aug 2023 deep-depression case study
the model captured the initial heat loss but recovered too quickly. The hypothesis is that this
is a data limitation, not an architecture limitation.

## 2. Hypotheses

- **H1.** Training on multiple years (several monsoon and cyclone seasons) reduces TCHP error
  against ARGO on an unseen year, compared with training on the same 3 months as v2.1.
- **H2.** With multi-year data, the cyclone-aware physics loss (`configs/main.yaml`) gives lower
  TCHP error against ARGO than the no-physics ablation (`configs/ablation_nophys.yaml`).

## 3. Data and split, fixed now

- Region and inputs as in v2.1 (METHODOLOGY §1). Period: the most recent consecutive years for
  which all inputs and GLORYS overlap, determined before training.
- **Test year:** the last full year. **Validation:** the last 2 months before the test year.
  **Training:** everything earlier. No random splitting.
- The test year is not opened (no plots, no metrics) until all training runs are finished.

## 4. Primary outcome, declared in advance

**TCHP RMSE against ARGO profiles in the test year** (profiles from ≤ 10 m to ≥ 200 m), main
model, mean over seeds 0, 1, 2.

## 5. Success criteria, declared in advance

- **H1 supported** if the multi-year main model's primary outcome is lower than the 3-month main
  model's on the same test year by more than 2 × the pooled seed standard deviation.
- **H2 supported** if main < no-physics on the primary outcome by more than 2 × the pooled seed
  standard deviation **and** the sign agrees on TCHP RMSE vs GLORYS.
- Anything smaller is reported as "no detectable difference".

## 6. Secondary outcomes (reported, not used to decide H1/H2)

Temperature RMSE per depth vs GLORYS and ARGO; D26, D20 and MLD errors; 90 % interval coverage
against GLORYS and ARGO; case studies of every IMD-designated depression or stronger system in
the Bay of Bengal during the test year (list fixed from IMD reports before evaluation).

## 7. Commitments

- All four configurations and all three seeds are reported; no run is dropped.
- No hyperparameter is changed after the test year is opened. Configs are frozen at the commit
  that adds this file's first amendment-free version.
- If a bug is found after opening the test year, the fix and both sets of numbers are reported.

## Amendments

*(none)*
