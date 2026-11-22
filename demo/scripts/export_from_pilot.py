"""Export the chat demo's data file from the pilot study's retrieval runs.

Provenance script, kept for transparency: it reads the pilot artifacts (per-query
BM25 + BGE-small candidate lists from feasibility check D, the BEIR corpora and
the SQuAD v1.1 dev set), which are not in this repo.

For each knowledge base it:
  1. tunes the blend weight and the score cutoff with calibre_rag on half of the
     test questions (cutoff chosen to match top-5's recall with fewer passages),
  2. runs calibre_rag v0 on the other half and stores, per question, the ranked
     passages, how many top-5 and Calibre send, and the relevance labels.

Usage: python demo/scripts/export_from_pilot.py PILOT_DIR demo/data/demo_data.js
"""
import json, random, re, sys

import numpy as np

from calibre_rag import Calibre, Hit
from calibre_rag.cutoff import TopK, fit_score_threshold
from calibre_rag.fusion import convex_combination, tune_alpha

PILOT, OUT = sys.argv[1], sys.argv[2]
TOPN = 30            # each retriever's top-30 is fused
KEEP = 10            # ranked passages stored per question
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

out = {"github": "https://github.com/v5run/Calibre", "topk": 5, "kbs": []}
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

    # 1. tune on one half, with the library
    L = {qi: lists_for(z, qi) for qi in test}
    runs = {str(q): (L[q][0], L[q][1]) for q in tune_q}
    qrels = {str(q): L[q][2] for q in tune_q}
    alpha, _ = tune_alpha(runs, qrels)
    policy, _ = fit_score_threshold({q: convex_combination(*runs[q], alpha=alpha) for q in runs}, qrels, match_topk=5)
    calibre = Calibre(alpha=alpha, cutoff=policy)
    topk = TopK(out["topk"])

    # 2. run v0 on the other half
    bank, used = [], set()
    for qi in eval_q:
        text = str(z["qtext"][qi]).strip()
        if len(text) > 180:
            continue
        b, dn, rel, tok = L[qi]
        res = calibre.retrieve(b, dn)
        ranked = res.candidates[:KEEP]
        k_top = topk.select(res.candidates)
        rel_top20 = sum(1 for h in res.candidates if h.doc_id in rel)
        item = {
            "q": text,
            "docs": [h.doc_id for h in ranked],
            "k_cal": res.cutoff_k, "k_top": k_top,
            "rel": {d: g for d, g in rel.items() if d in {h.doc_id for h in ranked}},
            "rel_top20": rel_top20,
        }
        if kb["id"] == "squad":
            item["answers"] = squad_answers.get(str(z["qids"][qi]), [])
        # tokens for the passages each method sends
        item["tok_cal"] = sum(tok[h.doc_id] for h in res.candidates[: res.cutoff_k])
        item["tok_top"] = sum(tok[h.doc_id] for h in res.candidates[:k_top])
        # what the question demonstrates (used to pick suggestions)
        cal_rel = sum(1 for h in res.candidates[: res.cutoff_k] if h.doc_id in rel)
        top_rel = sum(1 for h in res.candidates[:k_top] if h.doc_id in rel)
        item["kind"] = ("missed" if rel_top20 == 0 else "several" if rel_top20 >= 3
                        else "trimmed" if cal_rel < top_rel else "clear")
        bank.append(item)
        used |= {h.doc_id for h in ranked}

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
    want = {"clear": 4, "several": 2, "trimmed": 1, "missed": 1}
    sugg = []
    for kind, n in want.items():
        # skip questions that only make sense with their paragraph ("What is located within this district?")
        vague = re.compile(r"\b(this|these|those|he|she|they|it|his|her|their|its)\b", re.I)
        ok = lambda it: (it["kind"] == kind and (kind != "clear" or it["k_cal"] <= 2)
                         and not vague.search(it["q"]))
        pool = sorted([i for i, it in enumerate(bank) if ok(it)], key=lambda i: len(bank[i]["q"]))[: n * 3]
        random.Random(SEED + n).shuffle(pool)
        sugg += pool[:n]

    out["kbs"].append({
        "id": kb["id"], "name": kb["name"], "source": kb["source"], "placeholder": kb["placeholder"],
        "alpha": alpha, "tau": round(policy.tau, 4), "bank": bank, "passages": passages, "suggested": sugg,
    })
    kinds = {k: sum(1 for it in bank if it["kind"] == k) for k in want}
    print(kb["id"], "alpha", alpha, "tau", round(policy.tau, 3), "bank", len(bank), kinds, "passages", len(passages))

payload = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
with open(OUT, "w", encoding="utf-8") as fh:
    # a .js file loads with a plain <script> tag, so the demo also works when opened from disk
    fh.write("window.CALIBRE_DEMO = " + payload + ";\n")
