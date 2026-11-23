"""v1 + v2 in one script, on synthetic data (no downloads).

    python examples/try_guarantee.py

1. Fit a Calibrator on labelled queries        -> P(relevant) per passage (v1)
2. Fit a recall guarantee on other queries     -> cutoff with E[recall] >= 90% (v2)
3. Check both on queries neither step has seen
"""
import numpy as np

from calibre_rag import Calibre, Hit
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import TopK, fit_recall_guarantee, realised_recall


def make_runs(n_queries, seed):
    """Fake BM25 + dense results: relevant docs score higher in both, with noise."""
    rng = np.random.default_rng(seed)
    runs, qrels = {}, {}
    for q in range(n_queries):
        qid, n = f"q{seed}_{q}", 30
        rel = rng.choice(n, size=rng.integers(0, 4), replace=False)
        is_rel = np.isin(np.arange(n), rel)
        bm = rng.normal(5, 2, n) + 4 * is_rel * rng.uniform(0.3, 1.5, n)
        de = rng.normal(0.4, 0.1, n) + 0.2 * is_rel * rng.uniform(0.3, 1.5, n)
        ids = [f"{qid}_d{i}" for i in range(n)]
        runs[qid] = ([Hit(ids[i], float(bm[i])) for i in np.argsort(-bm)[:20]],
                     [Hit(ids[i], float(de[i])) for i in np.argsort(-de)[:20]])
        qrels[qid] = {ids[i]: 1 for i in rel}
    return runs, qrels


train, qrels = make_runs(400, seed=1)
calib, q2 = make_runs(400, seed=2); qrels.update(q2)
test, q3 = make_runs(800, seed=3); qrels.update(q3)

# v1: calibrated probabilities
cal = Calibrator(alpha=0.5).fit(train, qrels)
print(f"v1  ECE on unseen queries: {cal.ece(test, qrels):.3f}  (0 = probabilities are honest)")

# v2: cutoffs with a 90% recall guarantee, fitted on separate calibration queries.
#   on average:       E[recall] >= 0.9 over calibration draws (one draw can land a bit under)
#   high-confidence:  recall >= 0.9 with probability 0.9 over calibration draws (sends more)
C = cal.predict_runs(calib)
policy = fit_recall_guarantee(C, qrels, target=0.9)
strict = fit_recall_guarantee(C, qrels, target=0.9, delta=0.1)
print(f"v2  fitted on {policy.report['n_calibration_queries']} calibration queries")

T = cal.predict_runs(test)
for name, pol in [("fixed top-5", TopK(5)), ("fixed top-10", TopK(10)),
                  ("guarantee 90% (on average)", policy), ("guarantee 90% (high-confidence)", strict)]:
    r = realised_recall(pol, T, qrels)
    print(f"    {name:32s} recall {r['recall']:.3f}  passages {r['mean_k']:.2f}")

# One query through the full pipeline
calibre = Calibre(level="guaranteed", calibrator=cal, cutoff=policy)
res = calibre.retrieve(*test["q3_0"])
print(f"\nOne query: sends {res.cutoff_k} passages, expects {res.expected_relevant:.2f} relevant, "
      f"nothing_relevant={res.nothing_relevant}")
for h in res.hits:
    print(f"    {h.doc_id:10s} fused {h.fused_score:.2f}  P(relevant) {h.prob:.2f}")
