"""Fit cutoff parameters on training queries ("deployable" protocol from
feasibility check D: pick tau on train to match top-k's train recall)."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..eval.metrics import selection_recall
from ..types import ScoredHit
from .policies import ScoreThreshold, TopK

FusedRuns = Mapping[str, Sequence[ScoredHit]]   # qid -> fused, sorted hits
Qrels = Mapping[str, Mapping[str, float]]


def _mean_recall(fused: FusedRuns, qrels: Qrels, policy, depth: int) -> tuple[float, float]:
    rec, ks = [], []
    for q, hits in fused.items():
        if q not in qrels:
            continue
        k = policy.select(hits)
        rec.append(selection_recall([h.doc_id for h in hits[:k]], qrels[q], depth))
        ks.append(k)
    return float(np.mean(rec)), float(np.mean(ks))


def fit_score_threshold(
    fused: FusedRuns,
    qrels: Qrels,
    match_topk: int = 5,
    k_min: int = 1,
    k_max: int = 20,
    n_grid: int = 200,
) -> tuple[ScoreThreshold, dict]:
    """Largest tau whose mean recall >= top-`match_topk` recall on these queries
    (largest tau = fewest passages). Returns the policy and a small report."""
    target, _ = _mean_recall(fused, qrels, TopK(match_topk), k_max)
    all_scores = np.array([h.fused_score for hits in fused.values() for h in hits[:k_max]])
    grid = np.unique(np.quantile(all_scores, np.linspace(0, 1, n_grid)))[::-1]
    for tau in grid:
        pol = ScoreThreshold(float(tau), k_min, k_max)
        rec, mean_k = _mean_recall(fused, qrels, pol, k_max)
        if rec >= target:
            return pol, {"target_recall": target, "recall": rec, "mean_k": mean_k, "match_topk": match_topk}
    pol = ScoreThreshold(float(grid[-1]), k_min, k_max)
    rec, mean_k = _mean_recall(fused, qrels, pol, k_max)
    return pol, {"target_recall": target, "recall": rec, "mean_k": mean_k, "match_topk": match_topk}
