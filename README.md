# Calibre

**Calibrated hybrid retrieval that knows how much context is enough.**

Calibre is a drop-in layer between hybrid search (BM25 + dense) and the LLM in a RAG pipeline or agent. Instead of always sending a fixed top-k, it:

1. **Fuses** BM25 and dense scores with a tuned fixed blend (matched or beat RRF on most pilot datasets).
2. **Calibrates** each passage into a probability of relevance *(v1)*.
3. **Cuts off** the list adaptively, sending 1 passage for an easy question and 8 for a hard one.
4. **Guarantees** a recall target on that cutoff *(v2)*, even after a dataset or model change *(v3)*.
5. **Flags** when nothing retrieved is relevant, so an agent can retry, widen or abstain *(v1, v4)*.

## Versions

Each version adds one capability, ships on its own, and supports one claim.

| Version | Level | Adds | Status |
|---|---|---|---|
| v0 | `baseline` | Tuned fixed blend, raw-score cutoff, top-k fallback | **Implemented** |
| v1 | `calibrated` | P(relevant) per passage, probability cutoff, `nothing_relevant` flag | Planned |
| v2 | `guaranteed` | Recall guarantee on the cutoff (expected recall / conformal) | Planned |
| v3 | `portable` | Guarantees that survive dataset / embedding-model changes | Planned (research core) |
| v4 | agentic | Search tool for agents (smolagents / MCP) | Planned |

Each version has a folder in [`experiments/`](experiments/) with its hypothesis, plan and results.

## Quickstart (v0)

### 1. Install (Windows PowerShell)

```powershell
cd C:\Users\v5run\github\Calibre
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. On macOS/Linux, activate with `source .venv/bin/activate`.

### 2. Run the tests

```powershell
pytest
```

### 3. Try the demo

```powershell
python examples/try_calibre.py
```

It blends a small set of fake BM25 and dense results, then shows that a score cutoff (τ = 0.7) sends only the clearly relevant passage, while a fixed top-5 sends everything:

```
Fused ranking: [('doc1', 0.96), ('doc4', 0.68), ('doc7', 0.29), ('doc9', 0.21), ('doc2', 0.0)]
Top-5 sends:   ['doc1', 'doc4', 'doc7', 'doc9', 'doc2']
Cutoff sends:  ['doc1'] (1 passage)
```

### 4. Use it in code

```python
from calibre_rag import Calibre, Hit
from calibre_rag.cutoff import ScoreThreshold

bm25  = [Hit("doc1", 12.3), Hit("doc7", 9.1), Hit("doc4", 3.2)]
dense = [Hit("doc4", 0.82), Hit("doc1", 0.79), Hit("doc9", 0.55)]

cal = Calibre(alpha=0.6, cutoff=ScoreThreshold(tau=0.7))
res = cal.retrieve(bm25, dense)
res.doc_ids, res.cutoff_k
```

With real search results, convert your BM25 and vector-store results into `Hit(doc_id, score)` lists and pass them to `retrieve()`.

### 5. Tune on labelled queries

```python
from calibre_rag.fusion import tune_alpha, convex_combination
from calibre_rag.cutoff import fit_score_threshold

alpha, curve = tune_alpha(train_runs, train_qrels)            # runs: qid -> (bm25_hits, dense_hits)
fused = {q: convex_combination(*train_runs[q], alpha=alpha) for q in train_runs}
policy, report = fit_score_threshold(fused, train_qrels, match_topk=5)   # fewest passages at top-5 recall
```

## Layout

```
calibre_rag/
  core.py        # Calibre entry point (versions = levels)
  types.py       # Hit, ScoredHit, Result
  fusion/        # v0: normalisation, convex combination, RRF, alpha tuning
  cutoff/        # v0: top-k, score threshold (+ fitting); v1–v2 policies stubbed
  calibrate/     # v1: calibrator (planned)
  shift/         # v3: shift-robust calibration and guarantees (planned)
  adapters/      # LangChain, LlamaIndex, Weaviate, Qdrant, smolagents/MCP (planned)
  eval/          # nDCG@k, recall@k, selection recall
examples/
  try_calibre.py # quick v0 demo
docs/figures/     # pilot figures used in this README
experiments/
  v0_baseline/  v1_calibrated/  v2_guaranteed/  v3_portable/  v4_agentic/
tests/
```

The import name is `calibre_rag` (distribution `calibre-rag`) to avoid clashing with the `calibre` e-book application's Python modules.

## Preliminary results (pilot study)

Before building this repo, a small pilot tested the core idea on five public retrieval datasets. These results are **preliminary**: they come from pilot code outside this repo and will be reproduced at scale by the experiments in [`experiments/`](experiments/).

**Setup:** BM25 + a small embedding model (MiniLM-L6 or BGE-small), datasets SciFact, NFCorpus, NPL, SQuAD (paragraphs) and CISI, each up to ~15k passages. Queries were split 50/50 into train and test, 10 random splits. No LLM was run; answer coverage on SQuAD was checked by string matching.

### Passages sent at the same recall as a fixed top-5

The cutoff threshold was tuned on training queries to match top-5's recall, then measured on held-out test queries (BGE-small; negative = fewer passages):

| Dataset | Raw-score cutoff (v0, implemented) | Calibrated cutoff (v1, planned) |
|---|---|---|
| SciFact | −28% | −34% |
| NFCorpus | −20% | −21% |
| SQuAD | −27% | −36% |
| CISI | +3% | −11% |
| NPL | **+15%** | **+34%** |

![Change in passages sent vs top-5](docs/figures/pilot_deployable_savings_vs_top5.png)

- On SciFact, NFCorpus and SQuAD, a cutoff sent 20–40% fewer passages than top-5 at equal recall. On SQuAD, the calibrated cutoff needed ~32% fewer tokens to keep the answer in the context.
- **Most of the saving comes from the cutoff itself** (v0); calibration (v1) added 5–15 more points on SciFact and SQuAD.
- **Dynamic cutoffs were worse on NPL**, where each query has many relevant documents, so a top-k fallback is needed. CISI results are noisy (76 queries).

<details>
<summary>Full passages-vs-recall curves for every method</summary>

![Passages sent vs recall](docs/figures/pilot_passages_vs_recall.png)

</details>

### Other findings

- **Fusion:** a tuned fixed blend (v0's default) beat RRF on SciFact, NFCorpus and SQuAD (up to +0.04 nDCG@10 on SQuAD) and was within ±0.015 on NPL and CISI.
- **Calibration** (v1): ECE ≈ 0.04 on held-out queries from the same dataset, vs ≈ 0.23 for raw fused scores, at ~0.13 ms per query.
- **Guarantees across datasets** (v3): a 90% recall guarantee held in-domain but landed anywhere from 38% to 100% on a dataset the calibrator hadn't seen; 25–50 labelled queries repaired most of the calibration. This is the main open research question.
- **Not tested yet:** stronger embedders, SPLADE, rerankers, larger corpora, and real LLM answers.

## Development

- Python ≥ 3.10; runtime dependency is NumPy only (scikit-learn arrives with v1).
- Experiment artifacts (`data/`, `cache/`, `runs/`, `*.pkl`) are git-ignored and regenerated by scripts.
- Tag each version on release (`v0.1`, `v1.0`, …).
