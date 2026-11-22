"""Per-query score normalisation.

Feasibility check C found min-max vs z-score is a small detail compared with
tuning the blend weight, so min-max is the default.
"""
from __future__ import annotations

import numpy as np


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def zscore(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    sd = scores.std()
    if sd < 1e-12:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / sd


NORMALIZERS = {"minmax": minmax, "zscore": zscore}
