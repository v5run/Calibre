import numpy as np
import pytest

from calibre_rag import Calibre
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import ExpectedRecall, RecallGuarantee, TopK, fit_recall_guarantee, realised_recall

from synthetic import make_runs, split


@pytest.fixture(scope="module")
def setup():
    runs, qrels = make_runs(1600, seed=7)
    cal = Calibrator(alpha=0.5).fit(split(runs, 0, 400), qrels)
    calib = cal.predict_runs(split(runs, 400, 800))
    test = cal.predict_runs(split(runs, 800, 1600))
    return cal, calib, test, qrels


@pytest.mark.parametrize("target", [0.8, 0.9, 0.95])
def test_guarantee_holds_on_held_out(setup, target):
    _, calib, test, qrels = setup
    pol = fit_recall_guarantee(calib, qrels, target=target)
    assert pol.report["feasible"]
    out = realised_recall(pol, test, qrels)
    assert out["recall"] >= target - 0.02          # finite-sample slack on 800 test queries


def test_guarantee_holds_on_average_over_splits():
    runs, qrels = make_runs(1200, seed=11)
    cal = Calibrator().fit(split(runs, 0, 300), qrels)
    rest = cal.predict_runs(split(runs, 300, 1200))
    keys = list(rest); rng = np.random.default_rng(0); recs = []
    for _ in range(30):
        rng.shuffle(keys)
        c, t = {k: rest[k] for k in keys[:300]}, {k: rest[k] for k in keys[300:]}
        recs.append(realised_recall(fit_recall_guarantee(c, qrels, target=0.9), t, qrels)["recall"])
    assert np.mean(recs) >= 0.9


def test_guarantee_uses_fewer_passages_than_k_max(setup):
    _, calib, test, qrels = setup
    pol = fit_recall_guarantee(calib, qrels, target=0.8)
    assert realised_recall(pol, test, qrels)["mean_k"] < realised_recall(TopK(20), test, qrels)["mean_k"]


def test_infeasible_target_keeps_k_max(setup):
    _, calib, _, qrels = setup
    small = dict(list(calib.items())[:5])
    pol = fit_recall_guarantee(small, qrels, target=0.99)
    assert not pol.report["feasible"] and pol.threshold == -np.inf


def test_fused_score_guarantee(setup):
    _, calib, test, qrels = setup
    pol = fit_recall_guarantee(calib, qrels, target=0.9, on="fused")
    assert realised_recall(pol, test, qrels)["recall"] >= 0.88


def test_expected_recall_policy(setup):
    _, _, test, _ = setup
    hits = test["q900"]
    k = ExpectedRecall(rho=0.9).select(hits)
    p = np.array([h.prob for h in hits])
    assert p[:k].sum() >= 0.9 * p.sum() - 1e-9
    assert k == 1 or p[: k - 1].sum() < 0.9 * p.sum()


def test_core_levels(setup):
    cal, calib, _, qrels = setup
    runs, _ = make_runs(1600, seed=7)
    r1 = Calibre(level="calibrated", calibrator=cal).retrieve(*runs["q1000"])
    assert r1.expected_relevant is not None and isinstance(r1.nothing_relevant, bool)
    pol = fit_recall_guarantee(calib, qrels, target=0.9)
    r2 = Calibre(level="guaranteed", calibrator=cal, cutoff=pol).retrieve(*runs["q1000"])
    assert 1 <= r2.cutoff_k <= 20 and r2.level == "guaranteed"
    with pytest.raises(ValueError):
        Calibre(level="guaranteed", calibrator=cal)        # needs a fitted guarantee


def test_high_confidence_mode_is_more_conservative(setup):
    _, calib, test, qrels = setup
    exp = fit_recall_guarantee(calib, qrels, target=0.9)
    hc = fit_recall_guarantee(calib, qrels, target=0.9, delta=0.1)
    assert hc.threshold <= exp.threshold
    r = realised_recall(hc, test, qrels)
    assert r["recall"] >= 0.9 and r["mean_k"] >= realised_recall(exp, test, qrels)["mean_k"]
