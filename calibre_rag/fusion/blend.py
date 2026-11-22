"""Hybrid fusion: convex combination (the default) and RRF (a baseline).

Feasibility check C: a tuned fixed convex combination beat RRF on every
collection tested, and per-query blend weights mostly fit noise. So v0 uses
one global alpha, tuned once.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import Hit, ScoredHit
from .normalize import NORMALIZERS


def _normalised(hits: Sequence[Hit], normalizer: str) -> dict[str, float]:
    if not hits:
        return {}
    norm = NORMALIZERS[normalizer]
    vals = norm(np.array([h.score for h in hits], dtype=float))
    return {h.doc_id: float(v) for h, v in zip(hits, vals)}


def convex_combination(
    bm25_hits: Sequence[Hit],
    dense_hits: Sequence[Hit],
    alpha: float = 0.5,
    normalizer: str = "minmax",
) -> list[ScoredHit]:
    """fused = alpha * norm(dense) + (1 - alpha) * norm(bm25).

    alpha = 1.0 is dense-only, alpha = 0.0 is BM25-only. A document missing
    from one list gets that list's minimum normalised score (0 for min-max).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    nb = _normalised(bm25_hits, normalizer)
    nd = _normalised(dense_hits, normalizer)
    floor_b = min(nb.values()) if nb else 0.0
    floor_d = min(nd.values()) if nd else 0.0
    raw_b = {h.doc_id: h.score for h in bm25_hits}
    raw_d = {h.doc_id: h.score for h in dense_hits}

    out = []
    for doc_id in set(nb) | set(nd):
        fused = alpha * nd.get(doc_id, floor_d) + (1.0 - alpha) * nb.get(doc_id, floor_b)
        out.append(ScoredHit(doc_id, fused, raw_b.get(doc_id), raw_d.get(doc_id)))
    out.sort(key=lambda h: (-h.fused_score, h.doc_id))
    return out


def rrf(bm25_hits: Sequence[Hit], dense_hits: Sequence[Hit], k: int = 60) -> list[ScoredHit]:
    """Reciprocal rank fusion (baseline). Ranks are taken from input order."""
    scores: dict[str, float] = {}
    for hits in (bm25_hits, dense_hits):
        for rank, h in enumerate(hits, start=1):
            scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (k + rank)
    raw_b = {h.doc_id: h.score for h in bm25_hits}
    raw_d = {h.doc_id: h.score for h in dense_hits}
    out = [ScoredHit(d, s, raw_b.get(d), raw_d.get(d)) for d, s in scores.items()]
    out.sort(key=lambda h: (-h.fused_score, h.doc_id))
    return out
