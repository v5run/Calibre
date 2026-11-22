"""Core data types shared across all Calibre versions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Hit:
    """One result from a single retriever (BM25 or dense)."""
    doc_id: str
    score: float


@dataclass
class ScoredHit:
    """A fused candidate. `prob` is filled from v1 ("calibrated") onward."""
    doc_id: str
    fused_score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    prob: Optional[float] = None


@dataclass
class Result:
    """What Calibre returns for one query."""
    hits: list[ScoredHit]                    # selected passages, best first
    cutoff_k: int                            # how many were kept
    level: str                               # which version produced this
    candidates: list[ScoredHit] = field(default_factory=list)  # full fused list before the cutoff
    expected_relevant: Optional[float] = None  # v1+: sum of probabilities
    nothing_relevant: Optional[bool] = None    # v1+: max prob below threshold
    diagnostics: dict = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]
