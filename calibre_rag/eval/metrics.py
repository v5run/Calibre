"""Ranking and cutoff metrics (BEIR-style linear-gain nDCG)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, float], k: int = 10) -> float:
    gains = [float(qrel.get(d, 0.0)) for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((float(v) for v in qrel.values() if v > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrel: Mapping[str, float], k: int) -> float:
    rel = {d for d, v in qrel.items() if v > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def selection_recall(selected: Sequence[str], qrel: Mapping[str, float], depth: int = 20) -> float:
    """Recall of a variable-size selection, capped at `depth` relevant docs
    (matches feasibility check D: |selected ∩ rel| / min(|rel|, depth))."""
    rel = {d for d, v in qrel.items() if v > 0}
    if not rel:
        return 0.0
    return len(rel & set(selected)) / min(len(rel), depth)
