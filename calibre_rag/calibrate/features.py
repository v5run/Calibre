"""v1: per-candidate features describing where a passage sits in each result list
and how confident each list looks overall.

Min-max normalisation (v0) always gives the top result 1.0, even when every result
is weak. These features restore the missing information: absolute similarity,
z-scores, ranks, and the shape of each list.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..fusion.normalize import minmax, zscore
from ..types import Hit, ScoredHit

FEATURE_NAMES = (
    # per retriever: position relative to the other results (8)
    "bm25_minmax", "bm25_z", "bm25_logrank", "bm25_missing",
    "dense_minmax", "dense_z", "dense_logrank", "dense_missing",
    # absolute strength, comparable across queries (1)
    "dense_raw",
    # list shape: is this query's result list confident or flat? (6)
    "bm25_list_max", "bm25_list_std", "bm25_list_gap",
    "dense_list_max", "dense_list_std", "dense_list_gap",
    # query (1)
    "query_len",
)
N_FEATURES = len(FEATURE_NAMES)


def _list_stats(hits: Sequence[Hit]):
    if not hits:
        return {}, {}, {}, (0.0, 0.0, 0.0)
    scores = np.array([h.score for h in hits], dtype=float)
    mm, zz = minmax(scores), zscore(scores)
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(hits), dtype=int)
    rank[order] = np.arange(1, len(hits) + 1)
    ids = [h.doc_id for h in hits]
    top = np.sort(scores)[::-1]
    gap = float(top[0] - top[1]) if len(top) > 1 else 0.0
    return (dict(zip(ids, mm)), dict(zip(ids, zz)), dict(zip(ids, rank)),
            (float(top[0]), float(scores.std()), gap))


def candidate_features(
    candidates: Sequence[ScoredHit],
    bm25_hits: Sequence[Hit],
    dense_hits: Sequence[Hit],
    query: Optional[str] = None,
) -> np.ndarray:
    """Feature matrix, one row per candidate (same order as `candidates`)."""
    b_mm, b_z, b_rank, b_shape = _list_stats(bm25_hits)
    d_mm, d_z, d_rank, d_shape = _list_stats(dense_hits)
    nb, nd = max(len(bm25_hits), 1), max(len(dense_hits), 1)
    b_floor_z = min(b_z.values()) if b_z else 0.0
    d_floor_z = min(d_z.values()) if d_z else 0.0
    d_raw = {h.doc_id: h.score for h in dense_hits}
    d_raw_floor = min(d_raw.values()) if d_raw else 0.0
    qlen = float(len(query.split())) if query else 0.0

    rows = []
    for c in candidates:
        i = c.doc_id
        in_b, in_d = i in b_mm, i in d_mm
        rows.append([
            b_mm.get(i, 0.0), b_z.get(i, b_floor_z),
            math.log1p(b_rank.get(i, nb + 1)), 0.0 if in_b else 1.0,
            d_mm.get(i, 0.0), d_z.get(i, d_floor_z),
            math.log1p(d_rank.get(i, nd + 1)), 0.0 if in_d else 1.0,
            d_raw.get(i, d_raw_floor),
            *b_shape, *d_shape,
            qlen,
        ])
    return np.array(rows, dtype=float).reshape(len(rows), N_FEATURES)
