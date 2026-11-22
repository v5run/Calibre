import pytest

from calibre_rag import Calibre, Hit
from calibre_rag.cutoff import ScoreThreshold

BM25 = [Hit("a", 12.0), Hit("b", 8.0), Hit("c", 2.0)]
DENSE = [Hit("c", 0.9), Hit("b", 0.6), Hit("d", 0.3)]


def test_baseline_retrieve():
    res = Calibre(alpha=0.5).retrieve(BM25, DENSE)
    assert res.level == "baseline" and res.cutoff_k == 4 and len(res.candidates) == 4


def test_threshold_cutoff():
    res = Calibre(alpha=0.5, cutoff=ScoreThreshold(tau=0.6)).retrieve(BM25, DENSE)
    assert 1 <= res.cutoff_k < 4


def test_unimplemented_levels():
    for level in ("calibrated", "guaranteed", "portable"):
        with pytest.raises(NotImplementedError):
            Calibre(level=level)
    with pytest.raises(ValueError):
        Calibre(level="nope")
