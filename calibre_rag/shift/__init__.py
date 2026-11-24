"""v3 ("portable"): keep calibration and recall guarantees usable when the dataset
or the embedding model changes.

The problem. v1/v2 are fitted on one collection. On a new collection the score
scales, the number of relevant passages per query and the retriever mix all move,
so probabilities drift and a v2 cutoff can miss its recall target badly.

What is here:

    PortableCalibrator     trained on several source collections at once; every
                           feature is standardised *within its own collection*,
                           using unlabelled queries only. On a new collection,
                           .adapt(unlabelled_runs) re-standardises with no labels.
    importance_weights     label-free: how much each source calibration query looks
                           like the target's queries (domain classifier odds).
    fit_weighted_recall_guarantee
                           CRC with those weights (weighted conformal, label-free).
    fit_worst_source_guarantee
                           a cutoff that meets the target on every source
                           collection separately (label-free, conservative).
    few-shot               .recalibrate(k labelled target queries) re-fits Platt;
                           fit_recall_guarantee on those k queries gives an exact
                           guarantee for the target (costly for small k).

Guarantee status. Only the few-shot CRC route keeps v2's formal guarantee (it
calibrates on the target itself). The label-free routes are approximations:
weighted conformal assumes covariate shift (same P(relevant | features) across
collections), which retrieval collections only roughly satisfy. Measure them with
experiments/v3_portable before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from ..calibrate import Calibrator
from ..calibrate.model import LogisticModel, Platt, expected_calibration_error, fit_logistic, sigmoid
from ..cutoff.guarantee import _loss_curves, fit_recall_guarantee
from ..cutoff.policies import RecallGuarantee
from ..types import Hit, ScoredHit

Runs = Mapping[str, tuple[Sequence[Hit], Sequence[Hit]]]
Qrels = Mapping[str, Mapping[str, float]]
Source = tuple[Runs, Qrels]          # (runs, qrels) for one collection

__all__ = [
    "PortableCalibrator", "collection_stats", "importance_weights",
    "fit_weighted_recall_guarantee", "fit_worst_source_guarantee",
]


def collection_stats(base: Calibrator, runs: Runs, max_queries: int = 2000, seed: int = 0):
    """Mean and std of every calibrator input over a collection's candidates. Label-free."""
    qids = list(runs)
    if len(qids) > max_queries:
        qids = list(np.random.default_rng(seed).choice(qids, max_queries, replace=False))
    Xs = []
    for q in qids:
        b, d = runs[q]
        cands = base._candidates(b, d)
        if cands:
            Xs.append(base._matrix(cands, b, d))
    if not Xs:
        raise ValueError("no candidates in these runs")
    X = np.vstack(Xs)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


@dataclass
class PortableCalibrator:
    """Multi-source calibrator with per-collection feature standardisation.

        pc = PortableCalibrator(alpha=0.6).fit({"scifact": (runs, qrels), "nfcorpus": (...)})
        pc.adapt(new_runs)                  # label-free: standardise with the new collection
        hits = pc.predict(bm25, dense)      # ScoredHit with .prob
        pc.recalibrate(few_runs, few_qrels) # optional: Platt on 10-50 labelled target queries
    """
    alpha: float = 0.5
    normalizer: str = "minmax"
    depth: int = 20
    l2: float = 1.0
    model_: Optional[LogisticModel] = None
    stats_: dict = field(default_factory=dict)     # collection name -> (mu, sd)
    current: Optional[str] = None
    platt: Platt = field(default_factory=Platt)

    def __post_init__(self):
        self.base = Calibrator(alpha=self.alpha, normalizer=self.normalizer, depth=self.depth)

    def _norm(self, X: np.ndarray, name: str) -> np.ndarray:
        mu, sd = self.stats_[name]
        return (X - mu) / sd

    def fit(self, sources: Mapping[str, Source]) -> "PortableCalibrator":
        Xs, ys = [], []
        for name, (runs, qrels) in sources.items():
            self.stats_[name] = collection_stats(self.base, runs)
            X, y = self.base._dataset(runs, qrels)
            Xs.append(self._norm(X, name)); ys.append(y)
        self.model_ = LogisticModel(l2=self.l2).fit(np.vstack(Xs), np.concatenate(ys))
        self.platt = Platt()
        return self

    def adapt(self, runs: Runs, name: str = "target") -> "PortableCalibrator":
        """Point the calibrator at a collection using its unlabelled queries only."""
        self.stats_[name] = collection_stats(self.base, runs)
        self.current = name
        self.platt = Platt()
        return self

    def use(self, name: str) -> "PortableCalibrator":
        """Switch to a collection the calibrator already has statistics for."""
        if name not in self.stats_:
            raise KeyError(f"no statistics for collection {name!r}; call adapt() first")
        self.current = name
        return self

    def _probs(self, X: np.ndarray) -> np.ndarray:
        return self.platt(self.model_.predict_proba(self._norm(X, self.current)))

    def predict(self, bm25_hits: Sequence[Hit], dense_hits: Sequence[Hit], query: Optional[str] = None) -> list[ScoredHit]:
        if self.model_ is None or self.current is None:
            raise RuntimeError("call fit() and then adapt() or use()")
        cands = self.base._candidates(bm25_hits, dense_hits)
        if not cands:
            return []
        p = self._probs(self.base._matrix(cands, bm25_hits, dense_hits, query))
        return [replace(c, prob=float(pi)) for c, pi in zip(cands, p)]

    def predict_runs(self, runs: Runs, queries=None) -> dict[str, list[ScoredHit]]:
        return {q: self.predict(*runs[q], (queries or {}).get(q)) for q in runs}

    def recalibrate(self, runs: Runs, qrels: Qrels) -> "PortableCalibrator":
        """Few-shot: Platt re-fit on a handful of labelled queries from the current collection."""
        self.platt = Platt()
        X, y = self.base._dataset(runs, qrels)
        self.platt = Platt().fit(self.model_.predict_proba(self._norm(X, self.current)), y)
        return self

    def ece(self, runs: Runs, qrels: Qrels, n_bins: int = 15) -> float:
        X, y = self.base._dataset(runs, qrels)
        return expected_calibration_error(self._probs(X), y, n_bins)

    # Calibre(level="portable") uses the same interface as Calibrator
    @property
    def method(self):
        return "portable-logistic"


# ---------------------------------------------------------------- weighting

def _query_summary(hits: Sequence[ScoredHit], m: int = 10) -> np.ndarray:
    """Fixed-length description of one query's candidate list (label-free)."""
    p = np.array([h.prob for h in hits[:m]], dtype=float)
    f = np.array([h.fused_score for h in hits[:m]], dtype=float)
    pad = lambda a: np.concatenate([a, np.zeros(m - len(a))])
    p, f = pad(p), pad(f)
    return np.concatenate([p, f, [p.sum(), p.max(initial=0.0), len(hits) / 20.0]])


def importance_weights(source_cands: Mapping[str, Sequence[ScoredHit]],
                       target_cands: Mapping[str, Sequence[ScoredHit]],
                       l2: float = 1.0, clip: tuple[float, float] = (0.05, 20.0)):
    """w(q) ~ P(target | q) / P(source | q) for every source query, from a logistic
    domain classifier on query summaries. Label-free. Returns (weights for source
    queries, weights for target queries); both normalised to mean 1 over sources."""
    S = np.array([_query_summary(h) for h in source_cands.values()])
    T = np.array([_query_summary(h) for h in target_cands.values()])
    X = np.vstack([S, T]); y = np.r_[np.zeros(len(S)), np.ones(len(T))]
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1.0
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = fit_logistic(Z, y, l2=l2)
    odds = np.exp(np.clip(Z @ w, -30, 30)) * (len(S) / len(T))
    odds = np.clip(odds, *clip)
    ws, wt = odds[: len(S)], odds[len(S):]
    scale = ws.mean()
    return dict(zip(source_cands, ws / scale)), dict(zip(target_cands, wt / scale))


def fit_weighted_recall_guarantee(cands: Mapping[str, Sequence[ScoredHit]], qrels: Qrels,
                                  weights: Mapping[str, float], test_weight: float,
                                  target: float = 0.9, on: str = "prob",
                                  k_min: int = 1, k_max: int = 20) -> RecallGuarantee:
    """Weighted conformal risk control (Tibshirani et al. 2019; Angelopoulos et al. 2022):

        t_hat = max { t : sum_j w_j loss_j(t) / (W + w_test) + w_test / (W + w_test) <= 1 - target }

    with W = sum of calibration weights and w_test a weight representative of the
    target queries (e.g. a high quantile of importance_weights' target output)."""
    eps = 1.0 - target
    scores = np.array([(h.prob if on == "prob" else h.fused_score)
                       for hits in cands.values() for h in hits[:k_max]], dtype=float)
    grid = np.unique(np.concatenate([scores, [-np.inf, np.inf]]))
    kept_q = [q for q in cands if any(qrels.get(q, {}).get(h.doc_id, 0) > 0 for h in cands[q])]
    losses, kept = _loss_curves({q: cands[q] for q in kept_q}, qrels, on, k_min, k_max, grid)
    w = np.array([weights[q] for q in kept_q])
    W = w.sum()
    bound = (w @ losses) / (W + test_weight) + test_weight / (W + test_weight)
    ok = np.nonzero(bound <= eps)[0]
    feasible = ok.size > 0
    i = int(ok.max()) if feasible else 0
    report = {"n_calibration_queries": len(kept_q), "target": target, "mode": "weighted CRC (label-free)",
              "feasible": feasible, "effective_n": float(W ** 2 / (w ** 2).sum()),
              "test_weight": float(test_weight), "weighted_bound": float(bound[i])}
    return RecallGuarantee(threshold=float(grid[i]), target=target, on=on, k_min=k_min, k_max=k_max, report=report)


def fit_worst_source_guarantee(cands_by_source: Mapping[str, Mapping[str, Sequence[ScoredHit]]], qrels: Qrels,
                               target: float = 0.9, on: str = "prob", k_min: int = 1, k_max: int = 20,
                               delta: Optional[float] = None) -> RecallGuarantee:
    """The most aggressive threshold that meets the CRC bound on *every* source
    collection separately, i.e. the minimum of the per-source thresholds."""
    pols = {s: fit_recall_guarantee(c, qrels, target, on, k_min, k_max, delta) for s, c in cands_by_source.items()}
    worst = min(pols, key=lambda s: pols[s].threshold)
    report = {"mode": "worst source (label-free)", "target": target, "binding_source": worst,
              "per_source_threshold": {s: p.threshold for s, p in pols.items()},
              "feasible": all(p.report["feasible"] for p in pols.values())}
    return RecallGuarantee(threshold=pols[worst].threshold, target=target, on=on, k_min=k_min, k_max=k_max, report=report)
