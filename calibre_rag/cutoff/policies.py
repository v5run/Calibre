"""Cutoff policies.

v0: TopK, ScoreThreshold (raw fused score). Feasibility check D found a
    train-tuned raw-score threshold already saves 20-31% of passages vs top-5
    on 3 of 5 collections.
v1: ProbabilityThreshold (needs calibrated probabilities).
v2: ExpectedRecall, Conformal (recall guarantees).
"""
from __future__ import annotations

from dataclasses import dataclass
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
    """v2: smallest k with sum(p[:k]) >= rho * sum(p[:L])."""
    rho: float = 0.9

    def select(self, hits: Sequence[ScoredHit]) -> int:
        raise NotImplementedError("ExpectedRecall arrives in v2 ('guaranteed'); see experiments/v2_guaranteed")


@dataclass
class Conformal:
    """v2: split-conformal threshold with a marginal recall guarantee."""
    alpha: float = 0.1

    def select(self, hits: Sequence[ScoredHit]) -> int:
        raise NotImplementedError("Conformal arrives in v2 ('guaranteed'); see experiments/v2_guaranteed")
