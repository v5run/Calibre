# v1: Calibrated

**Hypothesis.** A lightweight calibrator over list features gives reliable P(relevant) (ECE ≈ 0.04 in-domain), ranks as well as CC-tuned, and a probability cutoff saves more passages than v0's raw-score cutoff.

**Plan**
- Calibrator: logistic regression over 16 list features, 5-fold cross-fitted; isotonic and per-retriever Platt variants.
- Outputs: per-passage probability, expected relevant count, `nothing_relevant` flag.
- Metrics: ECE, Brier, reliability diagrams; passages vs recall; AUROC of `nothing_relevant` for "no relevant passage in top-k".

**Done when:** calibrated vs raw-threshold savings table per dataset, with significance.

**Results:** _not run yet._
