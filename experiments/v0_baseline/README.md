# v0: Baseline

**Hypothesis.** A tuned fixed convex blend of BM25 and dense scores beats RRF, and a train-tuned raw-score cutoff sends fewer passages than fixed top-k at equal recall.

**Plan**
- Data: full BEIR (incl. MS MARCO dev subsample, NQ, HotpotQA, FiQA, SciFact, NFCorpus, ArguAna, Touché, TREC-COVID) + MIRACL-en.
- Retrievers: BM25 (Pyserini), SPLADE-v3, e5-base, bge-base/large, Qwen3-Embedding-0.6B.
- Methods: BM25, dense, RRF (k=60), CC α=0.5, CC-tuned; top-k (3/5/10), raw-score threshold, Weaviate autocut, adaptive-k.
- Metrics: nDCG@10, Recall@k, passages/tokens per query; paired tests with Holm correction.

**Done when:** a passages-vs-recall chart at scale, and a results table showing CC-tuned vs RRF per dataset.

**Pilot reference:** feasibility checks C and D (SciFact, NFCorpus, NPL, SQuAD, CISI with BM25 + MiniLM / BGE-small).

**Results:** _not run yet._
