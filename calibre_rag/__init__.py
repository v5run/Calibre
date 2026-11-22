"""Calibre: calibrated hybrid retrieval that knows how much context is enough.

Versions are feature levels of one package:
    v0 "baseline"   - tuned fixed blend + raw-score cutoff        (implemented)
    v1 "calibrated" - P(relevant) per passage + probability cutoff (planned)
    v2 "guaranteed" - recall guarantee on the cutoff               (planned)
    v3 "portable"   - guarantees that survive dataset/model shift  (planned)
    v4 "agentic"    - search tool for agents                       (planned)
"""
from .types import Hit, ScoredHit, Result
from .core import Calibre, LEVELS

__all__ = ["Calibre", "LEVELS", "Hit", "ScoredHit", "Result"]
__version__ = "0.1.0.dev0"
