"""v2: fit a cutoff with a recall guarantee via conformal risk control (CRC).

Setup. For calibration query j, let Rⱼ be the relevant passages among its ranked
candidates (top L), and keepⱼ(t) the passages kept by threshold t (clamped to
k_min..k_max). The loss

    lossⱼ(t) = 1 − |keepⱼ(t) ∩ Rⱼ| / |Rⱼ|          (1 − recall, between 0 and 1)

never decreases as t grows: a higher threshold keeps fewer passages.

CRC (Angelopoulos et al., 2022) picks the most aggressive threshold whose
finite-sample-corrected risk stays under the budget ε = 1 − target:

    t̂ = largest t with  (n / (n + 1)) · mean lossⱼ(t) + 1 / (n + 1) ≤ ε

Guarantee. If the calibration queries and a new query are exchangeable (drawn the
same way), then E[recall on the new query] ≥ target. It is an expected
(marginal) guarantee over queries; it covers relevant passages that retrieval put
in the top L (no cutoff can recover passages that were never retrieved); and it
is computed over queries with at least one relevant passage retrieved.
It does NOT hold across a change of dataset or model without recalibration (v3).

High-confidence mode (delta=0.1, say). The expected guarantee averages over which
calibration set you happened to draw: on one deployment, recall can land a little
under target. With delta set, the threshold is instead chosen with a Hoeffding
upper confidence bound (Learn-then-Test style, Angelopoulos et al., 2021):

    t_hat = max { t : mean_j loss_j(t) + sqrt( log(1/delta) / (2n) ) <= eps }

so that, with probability >= 1 - delta over the calibration draw, the expected
recall of the fitted cutoff is >= target. Costs more passages; shrinks as n grows.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..types import ScoredHit
from .policies import RecallGuarantee

CandRuns = Mapping[str, Sequence[ScoredHit]]   # qid -> ranked candidates (with .prob if on="prob")
Qrels = Mapping[str, Mapping[str, float]]


def _loss_curves(cands: CandRuns, qrels: Qrels, on: str, k_min: int, k_max: int, grid: np.ndarray):
    """Per-query loss at each grid threshold, plus kept counts. Shapes (n, len(grid))."""
    losses, kept = [], []
    for q, hits in cands.items():
        rel = qrels.get(q, {})
        top = list(hits[:k_max])
        is_rel = np.array([rel.get(h.doc_id, 0) > 0 for h in hits])
        n_rel = int(is_rel.sum())
        if n_rel == 0:
            continue                     # guarantee is over queries with >= 1 relevant retrieved
        s = np.array([(h.prob if on == "prob" else h.fused_score) for h in top], dtype=float)
        if on == "prob" and np.isnan(s).any():
            raise ValueError("on='prob' needs calibrated probabilities on every candidate")
        # number kept at each threshold, clamped like RecallGuarantee.select
        n_above = (s[None, :] >= grid[:, None]).sum(axis=1)
        k = np.clip(n_above, min(k_min, len(hits)), min(k_max, len(hits)))
        cum_rel = np.concatenate([[0], np.cumsum(is_rel[: len(top)])])
        losses.append(1.0 - cum_rel[k] / n_rel)
        kept.append(k)
    if not losses:
        raise ValueError("no calibration query has a relevant passage among its candidates")
    return np.array(losses), np.array(kept)


def fit_recall_guarantee(
    cands: CandRuns,
    qrels: Qrels,
    target: float = 0.9,
    on: str = "prob",
    k_min: int = 1,
    k_max: int = 20,
    delta: float | None = None,
) -> RecallGuarantee:
    """Most aggressive threshold with E[recall] >= target (CRC), or, with delta,
    with E[recall] >= target at confidence 1 - delta. Fit on held-out calibration
    queries that were NOT used to train the calibrator."""
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    eps = 1.0 - target
    scores = np.array([(h.prob if on == "prob" else h.fused_score)
                       for hits in cands.values() for h in hits[:k_max]], dtype=float)
    grid = np.unique(np.concatenate([scores, [-np.inf, np.inf]]))          # ascending
    losses, kept = _loss_curves(cands, qrels, on, k_min, k_max, grid)
    n = losses.shape[0]
    if delta is None:
        bound = n / (n + 1) * losses.mean(axis=0) + 1.0 / (n + 1)
    else:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        bound = losses.mean(axis=0) + np.sqrt(np.log(1.0 / delta) / (2 * n))
    ok = np.nonzero(bound <= eps)[0]
    feasible = ok.size > 0
    # mean loss is non-decreasing in t, so the feasible set is a prefix; take its last point
    i = int(ok.max()) if feasible else 0
    threshold = float(grid[i])
    report = {
        "n_calibration_queries": n,
        "target": target,
        "mode": "expected (CRC)" if delta is None else f"high-confidence (delta={delta})",
        "feasible": feasible,
        "crc_bound": float(bound[i]),
        "calibration_recall": float(1.0 - losses[:, i].mean()),
        "calibration_mean_k": float(kept[:, i].mean()),
    }
    if not feasible:
        report["note"] = (f"target {target} not reachable with k_max={k_max} and n={n} calibration queries; "
                          "keeping k_max. Lower the target, raise k_max, or add calibration queries.")
    return RecallGuarantee(threshold=threshold, target=target, on=on, k_min=k_min, k_max=k_max, report=report)


def realised_recall(policy, cands: CandRuns, qrels: Qrels) -> dict:
    """Evaluate any cutoff policy on held-out queries: mean per-query recall
    (queries with >= 1 relevant candidate) and mean passages kept."""
    rec, ks = [], []
    for q, hits in cands.items():
        rel = qrels.get(q, {})
        n_rel = sum(1 for h in hits if rel.get(h.doc_id, 0) > 0)
        k = policy.select(hits)
        ks.append(k)
        if n_rel:
            rec.append(sum(1 for h in hits[:k] if rel.get(h.doc_id, 0) > 0) / n_rel)
    return {"recall": float(np.mean(rec)) if rec else float("nan"), "mean_k": float(np.mean(ks)), "n": len(ks)}
