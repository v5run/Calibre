"""Validate v1 (calibration) and v2 (recall guarantee) on the pilot candidate caches.

Usage:  python validate_pilot.py CACHE_DIR [--seeds 20]
CACHE_DIR holds bge_<dataset>.npz from feasibility check D (BM25 + bge-small top-100 lists).

Protocol per dataset and seed: queries split 40% train (fit Calibrator) / 30% calibration
(fit the cutoff) / 30% test (report). Recall is over relevant passages among the top-20 candidates.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from calibre_rag import Hit
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import (ExpectedRecall, ProbabilityThreshold, TopK, fit_recall_guarantee,
                                fit_score_threshold, realised_recall)

ALPHA = {"squad": 0.5, "scifact": 0.7, "nfcorpus": 0.55}
TARGETS = (0.8, 0.9, 0.95)


def load(path):
    z = np.load(path, allow_pickle=True)
    p = z["qptr"]; runs, qrels, queries = {}, {}, {}
    for i, q in enumerate(z["qids"]):
        s = slice(p[i], p[i + 1])
        doc, bs, br, ds, dr, g = (z[k][s] for k in ("doc", "bs", "br", "ds", "dr", "g"))
        bm = sorted([(r, str(d), float(x)) for d, x, r in zip(doc, bs, br) if r >= 0])
        de = sorted([(r, str(d), float(x)) for d, x, r in zip(doc, ds, dr) if r >= 0])
        q = str(q)
        runs[q] = ([Hit(d, x) for _, d, x in bm], [Hit(d, x) for _, d, x in de])
        qrels[q] = {str(d): int(gg) for d, gg in zip(doc, g) if gg > 0}
        queries[q] = str(z["qtext"][i])
    return runs, qrels, queries


def sub(d, keys):
    return {k: d[k] for k in keys}


def one_seed(runs, qrels, queries, alpha, seed):
    keys = sorted(runs); rng = np.random.default_rng(seed); rng.shuffle(keys)
    n = len(keys); a, b = int(0.4 * n), int(0.7 * n)
    tr, ca, te = keys[:a], keys[a:b], keys[b:]
    cal = Calibrator(alpha=alpha).fit(sub(runs, tr), qrels, queries)
    C = cal.predict_runs(sub(runs, ca), queries)
    T = cal.predict_runs(sub(runs, te), queries)
    row = {"ece": cal.ece(sub(runs, te), qrels, queries)}
    top5 = realised_recall(TopK(5), T, qrels)
    row["top5"] = top5
    # v0 deployable: raw-score threshold matched to top-5 recall on the calibration split
    v0, _ = fit_score_threshold({q: v for q, v in C.items()}, qrels, match_topk=5)
    row["v0_threshold"] = realised_recall(v0, T, qrels)
    # v1: probability threshold, nothing-relevant flag
    row["v1_prob_0.25"] = realised_recall(ProbabilityThreshold(0.25), T, qrels)
    none_q = [q for q in T if not any(qrels[q].get(h.doc_id, 0) > 0 for h in T[q])]
    some_q = [q for q in T if q not in none_q]
    flag = lambda q: max(h.prob for h in T[q]) < 0.1
    row["nothing_flag"] = {"n_none": len(none_q),
                           "recall_on_none": float(np.mean([flag(q) for q in none_q])) if none_q else None,
                           "false_alarm": float(np.mean([flag(q) for q in some_q]))}
    # v2: nominal ExpectedRecall vs CRC guarantee
    for t in TARGETS:
        row[f"v2_expected_{t}"] = realised_recall(ExpectedRecall(rho=t), T, qrels)
        pol = fit_recall_guarantee(C, qrels, target=t)
        row[f"v2_guarantee_{t}"] = {**realised_recall(pol, T, qrels), "feasible": pol.report["feasible"]}
        hc = fit_recall_guarantee(C, qrels, target=t, delta=0.1)
        row[f"v2_highconf_{t}"] = {**realised_recall(hc, T, qrels), "feasible": hc.report["feasible"]}
    return row


def summarise(rows):
    out = {"ece": float(np.mean([r["ece"] for r in rows]))}
    for k in rows[0]:
        if k == "ece":
            continue
        if k == "nothing_flag":
            vals = [r[k] for r in rows]
            out[k] = {"n_none_avg": float(np.mean([v["n_none"] for v in vals])),
                      "recall_on_none": float(np.nanmean([v["recall_on_none"] if v["recall_on_none"] is not None else np.nan for v in vals])),
                      "false_alarm": float(np.mean([v["false_alarm"] for v in vals]))}
            continue
        rec = np.array([r[k]["recall"] for r in rows]); kk = np.array([r[k]["mean_k"] for r in rows])
        out[k] = {"recall": float(rec.mean()), "recall_min": float(rec.min()),
                  "frac_seeds_meeting_target": None, "mean_k": float(kk.mean())}
        if k.startswith("v2_"):
            t = float(k.rsplit("_", 1)[1])
            out[k]["frac_seeds_meeting_target"] = float((rec >= t).mean())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results.json"); args = ap.parse_args()
    res = {}
    for ds, alpha in ALPHA.items():
        runs, qrels, queries = load(Path(args.cache) / f"bge_{ds}.npz")
        rows = [one_seed(runs, qrels, queries, alpha, s) for s in range(args.seeds)]
        res[ds] = {"n_queries": len(runs), "alpha": alpha, **summarise(rows)}
        print(ds, json.dumps(res[ds], indent=1)[:4000], file=sys.stderr)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
