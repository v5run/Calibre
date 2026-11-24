"""Synthetic hybrid-retrieval data: relevant docs score higher in both retrievers (noisily)."""
import numpy as np

from calibre_rag.types import Hit


def make_runs(n_queries=300, n_docs=30, seed=0, bm_shift=0.0, bm_scale=1.0, dense_shift=0.0,
              max_rel=4, prefix="q"):
    """bm_shift/bm_scale/dense_shift/max_rel let tests build a *different collection*
    (other score scales and relevance rates) to test v3 shift handling."""
    rng = np.random.default_rng(seed)
    runs, qrels = {}, {}
    for q in range(n_queries):
        qid = f"{prefix}{q}"
        n_rel = rng.integers(0, max_rel)                    # 0-3 relevant; some queries have none
        rel = rng.choice(n_docs, size=n_rel, replace=False)
        is_rel = np.zeros(n_docs, bool); is_rel[rel] = True
        bm = bm_scale * (rng.normal(5, 2, n_docs) + 4 * is_rel * rng.uniform(0.3, 1.5, n_docs)) + bm_shift
        de = rng.normal(0.4, 0.1, n_docs) + 0.2 * is_rel * rng.uniform(0.3, 1.5, n_docs) + dense_shift
        ids = [f"{qid}_d{i}" for i in range(n_docs)]
        bm_order, de_order = np.argsort(-bm)[:20], np.argsort(-de)[:20]
        runs[qid] = ([Hit(ids[i], float(bm[i])) for i in bm_order],
                     [Hit(ids[i], float(de[i])) for i in de_order])
        qrels[qid] = {ids[i]: 1 for i in rel}
    return runs, qrels


def split(d, a, b):
    keys = sorted(d, key=lambda k: int(k[1:]))
    return {k: d[k] for k in keys[a:b]}
