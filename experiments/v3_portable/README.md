# v3: Portable (research core)

**Hypothesis.** With shift-aware methods, calibration and recall guarantees stay within ±0.05 of target on a new dataset or embedding model using ≤ 25 target labels.

**Plan**
- Leave-one-dataset-out and swap-the-encoder protocols.
- Methods: multi-source calibrators with label-free collection features; few-shot Platt re-fit (10/25/50/100); weighted conformal; worst-collection guarantees; prediction-powered inference with LLM-judged labels.
- Metrics: ECE and realised coverage on the target, vs number of target labels.

**Done when:** label-efficiency curves and a clear positive or negative result on guarantee transfer.

**Pilot reference:** in check D, zero-label transfer gave ECE 0.12–0.13 and 38–100% realised coverage at a 90% target; 25–50 labels repaired most of it.

**Results:** _not run yet._
