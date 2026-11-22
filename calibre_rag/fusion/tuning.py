"""Tune the global blend weight alpha on labelled queries."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..eval.metrics import ndcg_at_k
from ..types import Hit
from .blend import convex_combination

Runs = Mapping[str, tuple[Sequence[Hit], Sequence[Hit]]]  # qid -> (bm25_hits, dense_hits)
Qrels = Mapping[str, Mapping[str, float]]                 # qid -> {doc_id: graded relevance}


def tune_alpha(
    runs: Runs,
    qrels: Qrels,
    grid: Sequence[float] = tuple(np.round(np.linspace(0, 1, 21), 2)),
    normalizer: str = "minmax",
    k: int = 10,
) -> tuple[float, dict[float, float]]:
    """Return (best_alpha, {alpha: mean nDCG@k}). Ties break toward 0.5."""
    qids = [q for q in runs if q in qrels]
    if not qids:
        raise ValueError("no labelled queries overlap the runs")
    curve: dict[float, float] = {}
    for a in grid:
        vals = []
        for q in qids:
            fused = convex_combination(*runs[q], alpha=float(a), normalizer=normalizer)
            vals.append(ndcg_at_k([h.doc_id for h in fused], qrels[q], k))
        curve[float(a)] = float(np.mean(vals))
    best = max(curve, key=lambda a: (round(curve[a], 6), -abs(a - 0.5)))
    return best, curve
