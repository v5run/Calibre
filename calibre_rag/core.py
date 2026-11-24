"""The Calibre entry point. One class; versions are feature levels.

    level="baseline"    v0  tuned blend + top-k / score cutoff
    level="calibrated"  v1  + P(relevant) per passage, probability cutoff, nothing_relevant flag
    level="guaranteed"  v2  + cutoff with a recall guarantee (fit_recall_guarantee)
    level="portable"    v3  + calibrator that moves to a new dataset / embedding model (calibre_rag.shift)
"""
from __future__ import annotations

from typing import Optional, Sequence

from .calibrate import Calibrator
from .shift import PortableCalibrator
from .cutoff.policies import CutoffPolicy, ProbabilityThreshold, RecallGuarantee, TopK
from .fusion.blend import convex_combination
from .types import Hit, Result

LEVELS = ("baseline", "calibrated", "guaranteed", "portable")
_IMPLEMENTED = {"baseline", "calibrated", "guaranteed", "portable"}


class Calibre:
    """Hybrid retrieval layer between your retrievers and the LLM.

    v0:  Calibre(alpha=0.6, cutoff=ScoreThreshold(tau=0.72))
    v1:  Calibre(level="calibrated", calibrator=cal, cutoff=ProbabilityThreshold(0.25))
    v2:  Calibre(level="guaranteed", calibrator=cal, cutoff=fit_recall_guarantee(...))
    v3:  Calibre(level="portable", calibrator=PortableCalibrator(...).fit(sources).adapt(new_runs),
                 cutoff=fit_worst_source_guarantee(...) or fit_recall_guarantee(<few labelled target queries>))
    """

    def __init__(
        self,
        level: str = "baseline",
        alpha: float = 0.5,
        normalizer: str = "minmax",
        cutoff: Optional[CutoffPolicy] = None,
        depth: int = 20,
        calibrator: Optional[Calibrator | PortableCalibrator] = None,
        nothing_relevant_below: float = 0.1,
    ):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if level not in _IMPLEMENTED:
            raise NotImplementedError(
                f"level='{level}' is planned but not implemented yet "
                f"(implemented: {sorted(_IMPLEMENTED)}). See the roadmap in README.md."
            )
        if level in ("calibrated", "guaranteed", "portable"):
            if calibrator is None or calibrator.model_ is None:
                raise ValueError(f"level='{level}' needs a fitted Calibrator (calibrator=Calibrator(...).fit(...))")
        if level == "portable" and (not isinstance(calibrator, PortableCalibrator) or calibrator.current is None):
            raise ValueError("level='portable' needs PortableCalibrator(...).fit(sources).adapt(target_runs)")
        if level in ("calibrated", "guaranteed", "portable"):
            alpha, normalizer, depth = calibrator.alpha, calibrator.normalizer, calibrator.depth
        if level == "guaranteed" and not isinstance(cutoff, RecallGuarantee):
            raise ValueError("level='guaranteed' needs cutoff=fit_recall_guarantee(...)")
        self.level = level
        self.alpha = alpha
        self.normalizer = normalizer
        self.depth = depth
        self.calibrator = calibrator
        self.nothing_relevant_below = nothing_relevant_below
        default = ProbabilityThreshold(0.25) if level == "calibrated" else TopK(5)
        self.cutoff = cutoff or default

    def retrieve(self, bm25_hits: Sequence[Hit], dense_hits: Sequence[Hit], query: Optional[str] = None) -> Result:
        if self.calibrator is not None:
            cands = self.calibrator.predict(bm25_hits, dense_hits, query)
        else:
            cands = convex_combination(bm25_hits, dense_hits, self.alpha, self.normalizer)[: self.depth]
        k = self.cutoff.select(cands)
        res = Result(
            hits=cands[:k],
            cutoff_k=k,
            level=self.level,
            candidates=cands,
            diagnostics={"alpha": self.alpha, "normalizer": self.normalizer,
                         "cutoff": type(self.cutoff).__name__, "n_candidates": len(cands)},
        )
        if self.calibrator is not None and cands:
            probs = [h.prob for h in cands]
            res.expected_relevant = float(sum(probs))
            res.nothing_relevant = max(probs) < self.nothing_relevant_below
        return res
