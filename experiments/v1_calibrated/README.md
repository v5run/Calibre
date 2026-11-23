# v1: Calibrated

**Hypothesis.** A lightweight calibrator over list features gives reliable P(relevant) (ECE ≈ 0.04 in-domain), ranks as well as CC-tuned, and a probability cutoff saves more passages than v0's raw-score cutoff.

**Method** (`calibre_rag/calibrate/`)
- Candidates: v0's fused top-20.
- Features per candidate (16 + fused score): per retriever min-max score, z-score, log rank and missing flag; raw dense cosine; per-list max, std and top-1/top-2 gap; query length.
- Model: L2 logistic regression fitted by Newton/IRLS (NumPy only), or isotonic regression (PAV) on the fused score.
- `cross_fit` for out-of-fold probabilities, `recalibrate` for a Platt re-fit after a change, and `ece` to measure calibration.
- Outputs: `prob` per passage, `expected_relevant` = Σp, `nothing_relevant` = max p < 0.1.

**Results so far** (pilot caches, BGE-small + BM25, 20 random 40/30/30 splits; script: [`../v2_guaranteed/validate_pilot.py`](../v2_guaranteed/validate_pilot.py))

| Dataset | Queries | ECE (test) | `nothing_relevant`: caught | `nothing_relevant`: false alarms |
|---|---|---|---|---|
| SQuAD | 1,000 | 0.005 | 14% | 1.6% |
| SciFact | 1,109 | 0.007 | 18% | 1.3% |
| NFCorpus | 1,823 | 0.014 | 26% | 2.7% |

- **Calibration works in-domain.** ECE is 0.005–0.014, better than the pilot's 0.04 because these splits are larger.
- **The `nothing_relevant` flag is weak.** At a 0.1 threshold it rarely fires falsely, but it catches only 14–26% of queries with no relevant candidate. Per-passage probabilities alone don't separate "nothing here" well; that is v4's abstention work, which needs query-level features.
- A fixed probability threshold (0.25) is not a recall-controlled cutoff: recall swings by dataset (0.73 on SciFact, 0.85 on SQuAD). That is what v2 fixes.

**Still to do:** a reliability diagram, AUROC for `nothing_relevant`, isotonic vs logistic, and stronger embedders.
