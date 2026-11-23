"""How many fused passages to send to the LLM.

v0: TopK, ScoreThreshold (+ fit_score_threshold)
v1: ProbabilityThreshold
v2: ExpectedRecall, RecallGuarantee (+ fit_recall_guarantee, realised_recall)
"""
from .policies import (
    CutoffPolicy, TopK, ScoreThreshold,
    ProbabilityThreshold, ExpectedRecall, RecallGuarantee,
)
from .tuning import fit_score_threshold
from .guarantee import fit_recall_guarantee, realised_recall

__all__ = [
    "CutoffPolicy", "TopK", "ScoreThreshold",
    "ProbabilityThreshold", "ExpectedRecall", "RecallGuarantee",
    "fit_score_threshold", "fit_recall_guarantee", "realised_recall",
]
