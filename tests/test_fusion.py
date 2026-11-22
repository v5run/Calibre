import numpy as np
import pytest

from calibre_rag import Hit
from calibre_rag.fusion import convex_combination, minmax, rrf, tune_alpha, zscore

BM25 = [Hit("a", 12.0), Hit("b", 8.0), Hit("c", 2.0)]
DENSE = [Hit("c", 0.9), Hit("b", 0.6), Hit("d", 0.3)]


def test_minmax_range_and_constant():
    v = minmax(np.array([2.0, 4.0, 6.0]))
    assert v.min() == 0.0 and v.max() == 1.0
    assert np.all(minmax(np.array([3.0, 3.0])) == 1.0)


def test_zscore_mean_zero():
    v = zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(v.mean()) < 1e-12


def test_alpha_extremes_follow_single_retriever():
    assert [h.doc_id for h in convex_combination(BM25, DENSE, alpha=0.0)][:3] == ["a", "b", "c"]
    assert [h.doc_id for h in convex_combination(BM25, DENSE, alpha=1.0)][:2] == ["c", "b"]


def test_missing_doc_gets_list_floor():
    # "a" is absent from the dense list, so at alpha=1 it ties with the dense minimum ("d")
    fused = {h.doc_id: h.fused_score for h in convex_combination(BM25, DENSE, alpha=1.0)}
    assert fused["a"] == fused["d"] == 0.0


def test_union_and_raw_scores_kept():
    fused = {h.doc_id: h for h in convex_combination(BM25, DENSE, alpha=0.5)}
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused["a"].dense_score is None and fused["a"].bm25_score == 12.0
    assert fused["d"].bm25_score is None and fused["d"].dense_score == 0.3


def test_bad_alpha():
    with pytest.raises(ValueError):
        convex_combination(BM25, DENSE, alpha=1.5)


def test_rrf_rewards_agreement():
    assert rrf(BM25, DENSE)[0].doc_id in {"b", "c"}


def test_tune_alpha_finds_dense_when_dense_is_right():
    runs = {"q1": (BM25, DENSE)}
    qrels = {"q1": {"c": 1, "d": 1}}
    best, curve = tune_alpha(runs, qrels)
    assert best >= 0.5 and curve[1.0] >= curve[0.0]
