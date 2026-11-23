import pytest

from calibre_rag.cutoff import ExpectedRecall, ProbabilityThreshold, ScoreThreshold, TopK, fit_score_threshold
from calibre_rag.types import ScoredHit


def hits(scores, probs=None):
    probs = probs or [None] * len(scores)
    return [ScoredHit(f"d{i}", s, prob=p) for i, (s, p) in enumerate(zip(scores, probs))]


def test_topk_clamps():
    assert TopK(5).select(hits([0.9, 0.5])) == 2


def test_score_threshold_and_kmin():
    h = hits([0.9, 0.8, 0.3, 0.1])
    assert ScoreThreshold(tau=0.5).select(h) == 2
    assert ScoreThreshold(tau=0.99, k_min=1).select(h) == 1


def test_probability_threshold_requires_probs():
    with pytest.raises(ValueError):
        ProbabilityThreshold().select(hits([0.9, 0.8]))
    assert ProbabilityThreshold(tau=0.5).select(hits([0.9, 0.8, 0.1], [0.9, 0.6, 0.1])) == 2


def test_expected_recall_needs_probs():
    with pytest.raises(ValueError):
        ExpectedRecall().select(hits([0.9]))
    assert ExpectedRecall(rho=0.5).select(hits([0.9, 0.8, 0.1], [0.6, 0.3, 0.1])) == 1


def test_fit_score_threshold_matches_topk_recall_with_fewer_passages():
    fused = {
        "q1": hits([0.95, 0.40, 0.30, 0.20, 0.10, 0.05]),   # one clear winner
        "q2": hits([0.90, 0.88, 0.85, 0.20, 0.10, 0.05]),   # three good passages
    }
    qrels = {"q1": {"d0": 1}, "q2": {"d0": 1, "d1": 1, "d2": 1}}
    pol, rep = fit_score_threshold(fused, qrels, match_topk=5)
    assert rep["recall"] >= rep["target_recall"]
    assert rep["mean_k"] < 5
