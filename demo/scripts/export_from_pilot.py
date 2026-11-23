"""Export the chat demo's data file from the pilot study's retrieval runs.

Provenance script, kept for transparency: it reads the pilot artifacts (per-query
BM25 + BGE-small candidate lists from feasibility check D, the BEIR corpora and
the SQuAD v1.1 dev set), which are not in this repo.

For each knowledge base it:
  1. holds out half of the test questions for the demo; nothing below sees them,
  2. pools the dataset's train questions with the other half of the test questions,
     tunes the blend weight on the pool, trains the v1 Calibrator on 60% of it and
     fits v2 recall guarantees (80/90/95%) on the remaining 40%,
  3. runs calibre_rag on the demo questions and stores, per question, the ranked
     passages with P(relevant), how many passages each recall target and a fixed
     top-5 send, token counts and the relevance labels.

Usage: python demo/scripts/export_from_pilot.py PILOT_DIR demo/data/demo_data.js
"""
import json, random, re, sys

import numpy as np

from calibre_rag import Calibre, Hit
from calibre_rag.calibrate import Calibrator
from calibre_rag.cutoff import TopK, fit_recall_guarantee
from calibre_rag.fusion import tune_alpha

PILOT, OUT = sys.argv[1], sys.argv[2]
TOPN = 30            # each retriever's top-30 is fused
DEPTH = 20           # candidates per question that get a probability
TARGETS = (0.8, 0.9, 0.95)
DEFAULT_TARGET = "90"
NOTHING_BELOW = 0.1  # nothing_relevant flag: every passage below this probability
SEED = 13
BANK_MAX = 320       # questions per knowledge base
KBS = [
    dict(id="squad", name="Wikipedia", source="SQuAD v1.1", text_max=1200,
         placeholder="Ask a question about Wikipedia articles"),
    dict(id="scifact", name="Science papers", source="BEIR SciFact", text_max=600,
         placeholder="Check a scientific claim"),
    dict(id="nfcorpus", name="Nutrition research", source="BEIR NFCorpus", text_max=600,
         placeholder="Ask about a nutrition or health topic"),
]


def lists_for(z, qi):
    p = z["qptr"]; s = slice(p[qi], p[qi + 1])
    doc, bs, br, ds, dr, g = (z[k][s] for k in ("doc", "bs", "br", "ds", "dr", "g"))
    b = sorted((int(r), int(d), round(float(x), 4)) for d, x, r in zip(doc, bs, br) if 0 <= r < TOPN)
    dn = sorted((int(r), int(d), round(float(x), 4)) for d, x, r in zip(doc, ds, dr) if 0 <= r < TOPN)
    rel = {str(int(d)): int(gr) for d, gr in zip(doc, g) if gr > 0}
    tok = {str(int(d)): int(t) for d, t in zip(doc, z["tok"][s])}
    return [Hit(str(d), x) for _, d, x in b], [Hit(str(d), x) for _, d, x in dn], rel, tok


def split_title(t):
    """SQuAD docs are 'Title. paragraph'; skip initials/abbreviations ('Dwight D. Eisenhower')."""
    cut = 0
    while True:
        i = t.find(". ", cut)
        if i < 0:
            return "", t
        if re.search(r"(\b[A-Z]|\bSt|\bMr|\bDr|\bJr|\bvs)$", t[:i]):
            cut = i + 2
            continue
        return (t[:i], t[i + 2:]) if i < 120 else ("", t)


squad_answers = {}
for art in json.load(open(f"{PILOT}/../hubness/untrusted/squad/dev-v1.1.json"))["data"]:
    for para in art["paragraphs"]:
        for qa in para["qas"]:
            squad_answers[qa["id"]] = [a["text"] for a in qa["answers"]]

out = {"github": "https://github.com/v5run/Calibre", "topk": 5, "targets": [f"{round(t * 100)}" for t in TARGETS],
       "default_target": DEFAULT_TARGET, "nothing_below": NOTHING_BELOW, "kbs": []}
for kb in KBS:
    z = np.load(f"{PILOT}/cache/bge_{kb['id']}.npz")
    meta = json.load(open(f"{PILOT}/../hubness/data/{kb['id']}.json"))
    docs, doc_ids = meta["docs"], meta["doc_ids"]
    titles = {}
    try:
        with open(f"{PILOT}/../hybrid_fusion/data/{kb['id']}/corpus.jsonl") as fh:
            for line in fh:
                r = json.loads(line); titles[str(r["_id"])] = r.get("title", "")
    except FileNotFoundError:
        pass

    test = [i for i, s in enumerate(z["src"]) if s == "test"]
    random.Random(SEED).shuffle(test)
    tune_q, eval_q = test[: len(test) // 2], test[len(test) // 2:]
    pool = [i for i, s in enumerate(z["src"]) if s != "test"] + tune_q
    random.Random(SEED + 1).shuffle(pool)
    cut = int(0.6 * len(pool))
    fit_q, calib_q = pool[:cut], pool[cut:]

    # 1. tune, calibrate (v1) and fit guarantees (v2), never touching the demo questions
    L = {qi: lists_for(z, qi) for qi in range(len(z["src"]))}
    runs = {str(q): (L[q][0], L[q][1]) for q in pool}
    qrels = {str(q): L[q][2] for q in pool}
    alpha, _ = tune_alpha(runs, qrels)
    cal = Calibrator(alpha=alpha, depth=DEPTH).fit({str(q): runs[str(q)] for q in fit_q}, qrels)
    calib_cands = cal.predict_runs({str(q): runs[str(q)] for q in calib_q})
    policies = {f"{round(t * 100)}": fit_recall_guarantee(calib_cands, qrels, target=t) for t in TARGETS}
    calibres = {k: Calibre(level="guaranteed", calibrator=cal, cutoff=pol, nothing_relevant_below=NOTHING_BELOW)
                for k, pol in policies.items()}
    topk = TopK(out["topk"])

    # 2. run v1 + v2 on the demo questions
    bank = []
    for qi in eval_q:
        text = str(z["qtext"][qi]).strip()
        if len(text) > 180:
            continue
        b, dn, rel, tok = L[qi]
        res = {k: c.retrieve(b, dn) for k, c in calibres.items()}
        cands = res[DEFAULT_TARGET].candidates
        ks = {k: r.cutoff_k for k, r in res.items()}
        k_top = topk.select(cands)
        keep = max(max(ks.values()), k_top)
        ranked = cands[:keep]
        rel_top20 = sum(1 for h in cands if h.doc_id in rel)
        item = {
            "q": text,
            "docs": [h.doc_id for h in ranked],
            "p": [round(h.prob, 3) for h in ranked],
            "tk": [tok[h.doc_id] for h in ranked],
            "k": ks, "k_top": k_top,
            "nothing": res[DEFAULT_TARGET].nothing_relevant,
            "rel": {d: g for d, g in rel.items() if d in {h.doc_id for h in ranked}},
            "rel_top20": rel_top20,
        }
        if kb["id"] == "squad":
            item["answers"] = squad_answers.get(str(z["qids"][qi]), [])
        # what the question demonstrates at the default target (used to pick suggestions)
        k90 = ks[DEFAULT_TARGET]
        cal_rel = sum(1 for d in item["docs"][:k90] if d in rel)
        top_rel = sum(1 for d in item["docs"][:k_top] if d in rel)
        item["kind"] = ("flagged" if rel_top20 == 0 and item["nothing"] else "missed" if rel_top20 == 0
                        else "grew" if k90 > k_top and cal_rel > top_rel
                        else "trimmed" if cal_rel < top_rel else "several" if rel_top20 >= 3 else "clear")
        bank.append(item)

    random.Random(SEED).shuffle(bank)
    bank = bank[:BANK_MAX]
    used = {d for it in bank for d in it["docs"]}

    passages = {}
    for d in sorted(used, key=int):
        raw = docs[int(d)]
        t = titles.get(str(doc_ids[int(d)]))
        if t and raw.startswith(t):
            title, body = t, raw[len(t):].lstrip(" .")
        else:
            title, body = split_title(raw)
        if len(body) > kb["text_max"]:
            body = body[: kb["text_max"]].rsplit(" ", 1)[0] + "…"
        passages[d] = {"t": title, "x": body}

    # suggestions: mostly clear wins, plus honest cases
    want = {"clear": 3, "several": 1, "grew": 1, "flagged": 1, "trimmed": 1, "missed": 1}
    sugg = []
    for kind, n in want.items():
        # skip questions that only make sense with their paragraph ("What is located within this district?")
        vague = re.compile(r"\b(this|these|those|he|she|they|it|his|her|their|its)\b", re.I)
        ok = lambda it: (it["kind"] == kind and (kind != "clear" or it["k"][DEFAULT_TARGET] <= 2)
                         and not vague.search(it["q"]))
        pool = sorted([i for i, it in enumerate(bank) if ok(it)], key=lambda i: len(bank[i]["q"]))[: n * 3]
        random.Random(SEED + n).shuffle(pool)
        sugg += pool[:n]

    out["kbs"].append({
        "id": kb["id"], "name": kb["name"], "source": kb["source"], "placeholder": kb["placeholder"],
        "alpha": alpha, "bank": bank, "passages": passages, "suggested": sugg,
        "calibration": {k: {"threshold": round(p.threshold, 4), "n": p.report["n_calibration_queries"]}
                        for k, p in policies.items()},
    })
    kinds = {k: sum(1 for it in bank if it["kind"] == k) for k in want}
    mean_k = {k: round(sum(it["k"][k] for it in bank) / len(bank), 2) for k in policies}
    print(kb["id"], "alpha", alpha, "bank", len(bank), kinds, "passages", len(passages), "mean k", mean_k)

payload = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
with open(OUT, "w", encoding="utf-8") as fh:
    # a .js file loads with a plain <script> tag, so the demo also works when opened from disk
    fh.write("window.CALIBRE_DEMO = " + payload + ";\n")
