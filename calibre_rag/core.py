"""The Calibre entry point. One class; versions are feature levels."""
from __future__ import annotations

from typing import Optional, Sequence

from .cutoff.policies import CutoffPolicy, TopK
from .fusion.blend import convex_combination
from .types import Hit, Result

LEVELS = ("baseline", "calibrated", "guaranteed", "portable")
_IMPLEMENTED = {"baseline"}


class Calibre:
    """Hybrid retrieval layer between your retrievers and the LLM.

    Example (v0):
        cal = Calibre(alpha=0.6, cutoff=ScoreThreshold(tau=0.72))
        res = cal.retrieve(bm25_hits, dense_hits)
        res.hits, res.cutoff_k
    """

    def __init__(
        self,
        level: str = "baseline",
        alpha: float = 0.5,
        normalizer: str = "minmax",
        cutoff: Optional[CutoffPolicy] = None,
        depth: int = 20,
    ):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if level not in _IMPLEMENTED:
            raise NotImplementedError(
                f"level='{level}' is planned but not implemented yet "
                f"(implemented: {sorted(_IMPLEMENTED)}). See the roadmap in README.md."
            )
        self.level = level
        self.alpha = alpha
        self.normalizer = normalizer
        self.cutoff = cutoff or TopK(5)
        self.depth = depth

    def retrieve(self, bm25_hits: Sequence[Hit], dense_hits: Sequence[Hit]) -> Result:
        fused = convex_combination(bm25_hits, dense_hits, self.alpha, self.normalizer)[: self.depth]
        k = self.cutoff.select(fused)
        return Result(
            hits=fused[:k],
            cutoff_k=k,
            level=self.level,
            candidates=fused,
            diagnostics={"alpha": self.alpha, "normalizer": self.normalizer,
                         "cutoff": type(self.cutoff).__name__, "n_candidates": len(fused)},
        )
