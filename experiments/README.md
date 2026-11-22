# Experiments

One folder per version. Each README states the hypothesis, the plan, the "done when" bar and (once run) the results table and hero chart. Large artifacts go in git-ignored `data/`, `cache/`, `runs/`.

| Folder | Version | Timeline |
|---|---|---|
| `v0_baseline/` | Tuned blend + raw-score cutoff | Weeks 1–3 |
| `v1_calibrated/` | Calibrated probabilities + probability cutoff | Weeks 1–3 |
| `v2_guaranteed/` | Recall guarantees in-domain | Week 4 |
| `v3_portable/` | Guarantees under dataset / model shift | Weeks 5–7 |
| `v4_agentic/` | RAG and agent end-to-end evaluation | Weeks 8–9 |
