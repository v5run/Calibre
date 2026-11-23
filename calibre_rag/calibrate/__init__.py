"""v1 ("calibrated"): turn fused hybrid results into P(relevant) per passage.

    cal = Calibrator(alpha=0.6).fit(train_runs, train_qrels)
    hits = cal.predict(bm25_hits, dense_hits)      # fused top-L, each with .prob
    cal.recalibrate(new_runs, new_qrels)           # Platt re-fit after a dataset/model change

runs:  {qid: (bm25_hits, dense_hits)}   qrels: {qid: {doc_id: grade}}   queries: {qid: text} (optional)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from ..fusion.blend import convex_combination
from ..types import Hit, ScoredHit
from .features import FEATURE_NAMES, N_FEATURES, candidate_features
from .model import IsotonicModel, LogisticModel, Platt, expected_calibration_error, logit, sigmoid

Runs = Mapping[str, tuple[Sequence[Hit], Sequence[Hit]]]
Qrels = Mapping[str, Mapping[str, float]]

__all__ = ["Calibrator", "FEATURE_NAMES", "expected_calibration_error"]


@dataclass
class Calibrator:
    alpha: float = 0.5                 # blend weight used to rank candidates (from v0 tuning)
    normalizer: str = "minmax"
    method: str = "logistic"           # "logistic" (16 list features + fused score) | "isotonic" (fused score only)
    l2: float = 1.0
    depth: int = 20                    # candidates per query that get a probability
    platt: Platt = field(default_factory=Platt)
    model_: object = None

    # ---------- building blocks ----------
    def _candidates(self, bm25_hits, dense_hits):
        return convex_combination(bm25_hits, dense_hits, self.alpha, self.normalizer)[: self.depth]

    def _matrix(self, cands, bm25_hits, dense_hits, query=None) -> np.ndarray:
        X = candidate_features(cands, bm25_hits, dense_hits, query)
        fused = np.array([[c.fused_score] for c in cands], dtype=float).reshape(len(cands), 1)
        return np.hstack([X, fused])

    def _dataset(self, runs: Runs, qrels: Qrels, queries=None, qids=None):
        Xs, ys = [], []
        for q in (qids if qids is not None else runs):
            if q not in qrels:
                continue
            b, d = runs[q]
            cands = self._candidates(b, d)
            if not cands:
                continue
            Xs.append(self._matrix(cands, b, d, (queries or {}).get(q)))
            ys.append(np.array([1.0 if qrels[q].get(c.doc_id, 0) > 0 else 0.0 for c in cands]))
        if not Xs:
            raise ValueError("no labelled queries overlap the runs")
        return np.vstack(Xs), np.concatenate(ys)

    def _new_model(self):
        if self.method == "logistic":
            return LogisticModel(l2=self.l2)
        if self.method == "isotonic":
            return IsotonicModel(column=-1)
        raise ValueError("method must be 'logistic' or 'isotonic'")

    # ---------- public API ----------
    def fit(self, runs: Runs, qrels: Qrels, queries: Optional[Mapping[str, str]] = None) -> "Calibrator":
        X, y = self._dataset(runs, qrels, queries)
        self.model_ = self._new_model().fit(X, y)
        self.platt = Platt()
        return self

    def predict(self, bm25_hits: Sequence[Hit], dense_hits: Sequence[Hit], query: Optional[str] = None) -> list[ScoredHit]:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        cands = self._candidates(bm25_hits, dense_hits)
        if not cands:
            return []
        p = self.platt(self.model_.predict_proba(self._matrix(cands, bm25_hits, dense_hits, query)))
        return [replace(c, prob=float(pi)) for c, pi in zip(cands, p)]

    def predict_runs(self, runs: Runs, queries=None) -> dict[str, list[ScoredHit]]:
        return {q: self.predict(*runs[q], (queries or {}).get(q)) for q in runs}

    def cross_fit(self, runs: Runs, qrels: Qrels, queries=None, n_folds: int = 5, seed: int = 0) -> dict[str, list[ScoredHit]]:
        """Out-of-fold predictions: each query is scored by a model that never saw it.
        Use these (not in-sample predictions) to tune thresholds."""
        qids = [q for q in runs if q in qrels]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(qids))
        folds = [set(qids[i] for i in order[k::n_folds]) for k in range(n_folds)]
        out = {}
        for held in folds:
            sub = Calibrator(self.alpha, self.normalizer, self.method, self.l2, self.depth)
            sub.fit({q: runs[q] for q in qids if q not in held}, qrels, queries)
            for q in held:
                out[q] = sub.predict(*runs[q], (queries or {}).get(q))
        return out

    def recalibrate(self, runs: Runs, qrels: Qrels, queries=None) -> "Calibrator":
        """Platt re-fit (2 parameters) on a small labelled sample from the new dataset or model."""
        self.platt = Platt()
        X, y = self._dataset(runs, qrels, queries)
        self.platt = Platt().fit(self.model_.predict_proba(X), y)
        return self

    def ece(self, runs: Runs, qrels: Qrels, queries=None, n_bins: int = 15) -> float:
        X, y = self._dataset(runs, qrels, queries)
        return expected_calibration_error(self.platt(self.model_.predict_proba(X)), y, n_bins)
