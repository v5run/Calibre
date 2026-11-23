import numpy as np

from calibre_rag.calibrate import Calibrator
from calibre_rag.calibrate.model import expected_calibration_error, fit_logistic, pav, sigmoid

from synthetic import make_runs, split


def test_logistic_recovers_sign():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (rng.uniform(size=2000) < sigmoid(2 * X[:, 0] - X[:, 1])).astype(float)
    w = fit_logistic(np.c_[X, np.ones(2000)], y, l2=1e-3)
    assert w[0] > 1.5 and w[1] < -0.5


def test_pav_is_monotone():
    x = np.arange(10.0); y = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1], float)
    knots_x, knots_y = pav(x, y)
    assert np.all(np.diff(knots_y) >= -1e-12)


def test_calibrator_probs_are_calibrated_out_of_sample():
    runs, qrels = make_runs(600, seed=1)
    tr, te = split(runs, 0, 400), split(runs, 400, 600)
    cal = Calibrator(alpha=0.5).fit(tr, qrels)
    assert cal.ece(te, qrels) < 0.05
    hits = cal.predict(*te["q450"])
    assert all(0.0 <= h.prob <= 1.0 for h in hits)
    assert [h.fused_score for h in hits] == sorted([h.fused_score for h in hits], reverse=True)


def test_cross_fit_covers_every_query():
    runs, qrels = make_runs(100, seed=2)
    oof = Calibrator().cross_fit(runs, qrels, n_folds=5)
    assert set(oof) == set(runs) and all(h.prob is not None for v in oof.values() for h in v)


def test_isotonic_method():
    runs, qrels = make_runs(300, seed=3)
    cal = Calibrator(method="isotonic").fit(split(runs, 0, 200), qrels)
    assert cal.ece(split(runs, 200, 300), qrels) < 0.08


def test_ece_perfect_and_bad():
    p = np.full(1000, 0.3); y = (np.arange(1000) < 300).astype(float)
    assert expected_calibration_error(p, y) < 1e-9
    assert expected_calibration_error(np.full(1000, 0.9), y) > 0.5
