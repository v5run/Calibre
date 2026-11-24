"""v3: calibration and guarantees after a change of collection (synthetic)."""
import numpy as np
import pytest

from calibre_rag import Calibre
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import fit_recall_guarantee, realised_recall
from calibre_rag.shift import (PortableCalibrator, fit_weighted_recall_guarantee,
                               fit_worst_source_guarantee, importance_weights)

from synthetic import make_runs


def halves(runs, frac=0.5):
    keys = sorted(runs); c = int(frac * len(keys))
    return {k: runs[k] for k in keys[:c]}, {k: runs[k] for k in keys[c:]}


@pytest.fixture(scope="module")
def world():
    qrels = {}
    a, qa = make_runs(500, seed=1, prefix="a"); qrels.update(qa)
    b, qb = make_runs(500, seed=2, bm_scale=3.0, bm_shift=10.0, dense_shift=0.2, max_rel=6, prefix="b"); qrels.update(qb)
    # target: different score scales and fewer relevant passages
    t, qt = make_runs(1200, seed=3, bm_scale=0.4, bm_shift=-1.0, dense_shift=-0.15, max_rel=3, prefix="t"); qrels.update(qt)
    return {"a": a, "b": b}, t, qrels


def test_collection_standardisation_moves_with_the_collection(world):
    sources, t, qrels = world
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()})
    assert set(pc.stats_) == {"a", "b"}
    pool, test = halves(t)
    pc.adapt(pool)
    mu_t, _ = pc.stats_["target"]
    assert not np.allclose(mu_t, pc.stats_["a"][0])
    hits = pc.predict(*test[next(iter(test))])
    assert all(0 <= h.prob <= 1 for h in hits)


def test_few_shot_platt_repairs_calibration_on_a_new_collection(world):
    # Label-free standardisation alone is not guaranteed to beat naive transfer
    # (see experiments/v3_portable); a 50-query Platt re-fit should bring ECE down.
    sources, t, qrels = world
    pool, test = halves(t)
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()}).adapt(pool)
    before = pc.ece(test, qrels)
    pc.recalibrate(dict(list(pool.items())[:50]), qrels)
    assert pc.ece(test, qrels) < max(before, 0.03)


def test_few_shot_crc_on_target_holds(world):
    sources, t, qrels = world
    pool, test = halves(t)
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()}).adapt(pool)
    shots = dict(list(pool.items())[:300])
    pc.recalibrate(shots, qrels)
    pol = fit_recall_guarantee(pc.predict_runs(shots), qrels, target=0.85)
    assert realised_recall(pol, pc.predict_runs(test), qrels)["recall"] >= 0.83


def test_worst_source_is_the_most_conservative(world):
    sources, t, qrels = world
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()})
    cands = {}
    for s, r in sources.items():
        pc.use(s); cands[s] = pc.predict_runs(r)
    pol = fit_worst_source_guarantee(cands, qrels, target=0.9)
    assert pol.threshold == min(pol.report["per_source_threshold"].values())


def test_importance_weights_and_weighted_crc(world):
    sources, t, qrels = world
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()})
    src = {}
    for s, r in sources.items():
        pc.use(s); src.update(pc.predict_runs(r))
    pool, _ = halves(t)
    pc.adapt(pool)
    ws, wt = importance_weights(src, pc.predict_runs(pool))
    assert abs(np.mean(list(ws.values())) - 1.0) < 1e-9 and all(v > 0 for v in wt.values())
    pol = fit_weighted_recall_guarantee(src, qrels, ws, float(np.quantile(list(wt.values()), 0.9)), target=0.9)
    plain = fit_recall_guarantee(src, qrels, target=0.9)
    assert pol.report["effective_n"] <= len(src)
    # uniform weights reproduce plain CRC up to the test-weight term
    uni = fit_weighted_recall_guarantee(src, qrels, {q: 1.0 for q in src}, 1.0, target=0.9)
    assert uni.threshold == plain.threshold


def test_core_portable_level(world):
    sources, t, qrels = world
    pc = PortableCalibrator().fit({s: (r, qrels) for s, r in sources.items()})
    with pytest.raises(ValueError):
        Calibre(level="portable", calibrator=pc)          # not adapted to a collection yet
    pool, test = halves(t)
    pc.adapt(pool)
    pol = fit_recall_guarantee(pc.predict_runs(dict(list(pool.items())[:200])), qrels, target=0.8)
    res = Calibre(level="portable", calibrator=pc, cutoff=pol).retrieve(*test[next(iter(test))])
    assert res.level == "portable" and res.expected_relevant is not None
