"""Cutoff policies.

v0: TopK, ScoreThreshold (raw fused score). Feasibility check D found a
    train-tuned raw-score threshold already saves 20-31% of passages vs top-5
    on 3 of 5 collections.
v1: ProbabilityThreshold (needs calibrated probabilities).
v2: ExpectedRecall (nominal, from probabilities) and RecallGuarantee
    (threshold fitted with conformal risk control; see cutoff/guarantee.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..types import ScoredHit


class CutoffPolicy(Protocol):
    def select(self, hits: Sequence[ScoredHit]) -> int:
        """Return how many of the (sorted) hits to keep."""
        ...


@dataclass
class TopK:
    k: int = 5

    def select(self, hits: Sequence[ScoredHit]) -> int:
        return min(self.k, len(hits))


@dataclass
class ScoreThreshold:
    """Keep hits whose fused score is >= tau, clamped to [k_min, k_max]."""
    tau: float
    k_min: int = 1
    k_max: int = 20

    def select(self, hits: Sequence[ScoredHit]) -> int:
        n = sum(1 for h in hits[: self.k_max] if h.fused_score >= self.tau)
        return max(min(self.k_min, len(hits)), min(n, self.k_max, len(hits)))


@dataclass
class ProbabilityThreshold:
    """v1: keep hits with P(relevant) >= tau."""
    tau: float = 0.25
    k_min: int = 1
    k_max: int = 20

    def select(self, hits: Sequence[ScoredHit]) -> int:
        if any(h.prob is None for h in hits[: self.k_max]):
            raise ValueError("ProbabilityThreshold needs calibrated probabilities (level='calibrated')")
        n = sum(1 for h in hits[: self.k_max] if h.prob >= self.tau)
        return max(min(self.k_min, len(hits)), min(n, self.k_max, len(hits)))


@dataclass
class ExpectedRecall:
    """v2: smallest k with sum(p[:k]) >= rho * sum(p[:k_max]).

    The probabilities estimate how many relevant passages there are; this keeps
    enough of that expected mass. When probabilities are spread out (an uncertain
    ranking) it keeps more. Nominal only: the target holds if the probabilities are
    calibrated. For a distribution-free guarantee use RecallGuarantee.
    """
    rho: float = 0.9
    k_min: int = 1
    k_max: int = 20

    def select(self, hits: Sequence[ScoredHit]) -> int:
        top = hits[: self.k_max]
        if any(h.prob is None for h in top):
            raise ValueError("ExpectedRecall needs calibrated probabilities (level='calibrated' or 'guaranteed')")
        if not top:
            return 0
        total = sum(h.prob for h in top)
        k, acc = len(top), 0.0
        if total > 0:
            for i, h in enumerate(top, start=1):
                acc += h.prob
                if acc >= self.rho * total - 1e-12:
                    k = i
                    break
        return max(min(self.k_min, len(top)), min(k, len(top)))


@dataclass
class RecallGuarantee:
    """v2: keep passages whose score (probability or fused score) is >= threshold.

    Fitted with `fit_recall_guarantee`, which picks the most aggressive threshold
    whose conformal risk control bound keeps expected per-query recall >= target.
    """
    threshold: float
    target: float = 0.9
    on: str = "prob"                   # "prob" or "fused"
    k_min: int = 1
    k_max: int = 20
    report: dict = field(default_factory=dict)

    def _score(self, h: ScoredHit) -> float:
        if self.on == "prob":
            if h.prob is None:
                raise ValueError("RecallGuarantee(on='prob') needs calibrated probabilities")
            return h.prob
        return h.fused_score

    def select(self, hits: Sequence[ScoredHit]) -> int:
        n = sum(1 for h in hits[: self.k_max] if self._score(h) >= self.threshold)
        return max(min(self.k_min, len(hits)), min(n, self.k_max, len(hits)))
