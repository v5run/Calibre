"""v3: do calibration and recall guarantees survive a new dataset or embedding model?

Usage:  python run_shift.py CACHE_DIR [--seeds 10] [--target 0.9]
CACHE_DIR holds {bge,minilm}_<dataset>.npz from feasibility check D.

Two shifts:
  dataset  calibrate on four BGE collections, deploy on the fifth (leave one out)
  model    calibrate on all five BGE collections, deploy on the MiniLM version of one

Methods (recall target 0.9 unless --target):
  in-domain     v1 + v2 fitted on labelled target queries (the reference; needs ~500 labels)
  naive         v1 + v2 fitted on the pooled sources, used as-is on the target
  portable      PortableCalibrator (per-collection standardisation, label-free adapt) + CRC on sources
  worst-source  portable + the threshold that meets the target on every source
  weighted      portable + weighted CRC (importance weights from unlabelled target queries)
  fewshot-er-k  portable + Platt on k labelled target queries + ExpectedRecall(target)
  fewshot-crc-k portable + CRC fitted on k labelled target queries (exact guarantee)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from calibre_rag import Hit
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import ExpectedRecall, TopK, fit_recall_guarantee, realised_recall
from calibre_rag.fusion import tune_alpha
from calibre_rag.shift import (PortableCalibrator, fit_weighted_recall_guarantee,
                               fit_worst_source_guarantee, importance_weights)

DATASETS = ["squad", "scifact", "nfcorpus", "npl", "cisi"]
TARGETS = ["squad", "scifact", "nfcorpus"]      # npl/cisi are too small (93/76 queries) to evaluate on
KS = (10, 25, 50)


def load(path, prefix):
    z = np.load(path, allow_pickle=True)
    p = z["qptr"]; runs, qrels = {}, {}
    for i, q in enumerate(z["qids"]):
        s = slice(p[i], p[i + 1])
        doc, bs, br, ds, dr, g = (z[k][s] for k in ("doc", "bs", "br", "ds", "dr", "g"))
        bm = sorted([(r, str(d), float(x)) for d, x, r in zip(doc, bs, br) if r >= 0])
        de = sorted([(r, str(d), float(x)) for d, x, r in zip(doc, ds, dr) if r >= 0])
        qid = f"{prefix}:{q}"
        runs[qid] = ([Hit(d, x) for _, d, x in bm], [Hit(d, x) for _, d, x in de])
        qrels[qid] = {str(d): int(gg) for d, gg in zip(doc, g) if gg > 0}
    return runs, qrels


def sub(d, keys):
    return {k: d[k] for k in keys}


def split(keys, frac, seed):
    keys = sorted(keys); np.random.default_rng(seed).shuffle(keys)
    c = int(frac * len(keys))
    return keys[:c], keys[c:]


def evaluate(target_name, sources, target, qrels, target_rec, seeds):
    """sources: {name: runs}; target: runs. Returns per-method summary."""
    # ---- source side (fixed): 60% train the calibrator, 40% fit cutoffs
    tr, ca = {}, {}
    for s, runs in sources.items():
        tr[s], ca[s] = split(runs, 0.6, seed=123)
    pooled_tr = {q: sources[s][q] for s in sources for q in tr[s]}
    alpha, _ = tune_alpha(pooled_tr, qrels)

    naive = Calibrator(alpha=alpha).fit(pooled_tr, qrels)
    naive_pol = fit_recall_guarantee(naive.predict_runs({q: sources[s][q] for s in sources for q in ca[s]}),
                                     qrels, target=target_rec)

    pc = PortableCalibrator(alpha=alpha).fit({s: (sub(sources[s], tr[s]), qrels) for s in sources})
    src_cands = {}
    for s in sources:
        pc.use(s)
        src_cands[s] = pc.predict_runs(sub(sources[s], ca[s]))
    pooled_src = {q: h for s in src_cands for q, h in src_cands[s].items()}
    portable_pol = fit_recall_guarantee(pooled_src, qrels, target=target_rec)
    worst_pol = fit_worst_source_guarantee(src_cands, qrels, target=target_rec)

    rows = {}
    def add(name, r):
        rows.setdefault(name, []).append(r)

    for seed in range(seeds):
        pool, test = split(target, 0.5, seed)       # pool: unlabelled (+ few-shot labels); test: evaluation
        T_runs = sub(target, test)
        # reference: in-domain v1 + v2 on the (labelled) pool
        a, b = split(pool, 0.6, seed + 100)
        ind = Calibrator(alpha=alpha).fit(sub(target, a), qrels)
        ind_pol = fit_recall_guarantee(ind.predict_runs(sub(target, b)), qrels, target=target_rec)
        add("in-domain", {**realised_recall(ind_pol, ind.predict_runs(T_runs), qrels), "ece": ind.ece(T_runs, qrels)})
        # naive transfer
        add("naive", {**realised_recall(naive_pol, naive.predict_runs(T_runs), qrels), "ece": naive.ece(T_runs, qrels)})
        # label-free adaptation
        pc.adapt(sub(target, pool), name="target")
        T = pc.predict_runs(T_runs)
        ece_pc = pc.ece(T_runs, qrels)
        add("portable", {**realised_recall(portable_pol, T, qrels), "ece": ece_pc})
        add("worst-source", {**realised_recall(worst_pol, T, qrels), "ece": ece_pc})
        tgt_pool_cands = pc.predict_runs(sub(target, pool))
        w_src, w_tgt = importance_weights(pooled_src, tgt_pool_cands)
        wpol = fit_weighted_recall_guarantee(pooled_src, qrels, w_src, float(np.quantile(list(w_tgt.values()), 0.9)),
                                             target=target_rec)
        add("weighted", {**realised_recall(wpol, T, qrels), "ece": ece_pc, "eff_n": wpol.report["effective_n"]})
        # few-shot
        rng = np.random.default_rng(seed + 7)
        for k in KS:
            shots = list(rng.choice(sorted(pool), size=min(k, len(pool)), replace=False))
            pc.recalibrate(sub(target, shots), qrels)          # Platt on k labels (stats from the pool)
            T = pc.predict_runs(T_runs)
            add(f"fewshot-er-{k}", {**realised_recall(ExpectedRecall(rho=target_rec), T, qrels), "ece": pc.ece(T_runs, qrels)})
            crc = fit_recall_guarantee(pc.predict_runs(sub(target, shots)), qrels, target=target_rec)
            add(f"fewshot-crc-{k}", {**realised_recall(crc, T, qrels), "ece": pc.ece(T_runs, qrels),
                                     "feasible": crc.report["feasible"]})
        add("top-5", {**realised_recall(TopK(5), T, qrels), "ece": float("nan")})
        pc.platt = type(pc.platt)()                          # reset before the next seed

    out = {}
    for m, rs in rows.items():
        rec = np.array([r["recall"] for r in rs])
        out[m] = {"recall": float(rec.mean()), "recall_sd": float(rec.std()), "recall_min": float(rec.min()),
                  "abs_gap": float(np.abs(rec - target_rec).mean()),
                  "frac_meeting": float((rec >= target_rec - 1e-9).mean()),
                  "mean_k": float(np.mean([r["mean_k"] for r in rs])),
                  "ece": (float(np.nanmean([r["ece"] for r in rs])) if m != "top-5" else None)}
        if "feasible" in rs[0]:
            out[m]["frac_feasible"] = float(np.mean([r["feasible"] for r in rs]))
    out["_meta"] = {"alpha": alpha, "n_target": len(target), "sources": list(sources)}
    print(target_name, json.dumps({m: (round(v["recall"], 3), round(v["mean_k"], 2), v["ece"] and round(v["ece"], 3))
                                   for m, v in out.items() if m != "_meta"}), file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9); ap.add_argument("--out", default="results.json")
    ap.add_argument("--only", choices=["dataset", "model"], default=None)
    args = ap.parse_args()
    data, qrels = {}, {}
    for model in ("bge", "minilm"):
        for d in DATASETS:
            r, q = load(Path(args.cache) / f"{model}_{d}.npz", f"{model}-{d}")
            data[(model, d)] = r; qrels.update(q)
    res = {"target_recall": args.target, "dataset_shift": {}, "model_shift": {}}
    for t in TARGETS if args.only != "model" else []:
        srcs = {d: data[("bge", d)] for d in DATASETS if d != t}
        res["dataset_shift"][t] = evaluate(f"dataset:{t}", srcs, data[("bge", t)], qrels, args.target, args.seeds)
    for t in TARGETS if args.only != "dataset" else []:
        srcs = {d: data[("bge", d)] for d in DATASETS}
        res["model_shift"][t] = evaluate(f"model:{t}", srcs, data[("minilm", t)], qrels, args.target, args.seeds)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
