"""Quick demo of Calibre v0: blend BM25 + dense scores, then compare a fixed
top-k with a score cutoff.

Run from the repo root (after `pip install -e ".[dev]"`):
    python examples/try_calibre.py
"""
from calibre_rag import Calibre, Hit
from calibre_rag.cutoff import ScoreThreshold, TopK

# Pretend results from BM25 and from embedding search
bm25 = [Hit("doc1", 12.3), Hit("doc7", 9.1), Hit("doc4", 3.2), Hit("doc2", 1.0)]
dense = [Hit("doc4", 0.82), Hit("doc1", 0.79), Hit("doc9", 0.55), Hit("doc2", 0.40)]

fixed = Calibre(alpha=0.6, cutoff=TopK(5)).retrieve(bm25, dense)
cut = Calibre(alpha=0.6, cutoff=ScoreThreshold(tau=0.7)).retrieve(bm25, dense)

print("Fused ranking:", [(h.doc_id, round(h.fused_score, 2)) for h in fixed.candidates])
print("Top-5 sends:  ", fixed.doc_ids)
print("Cutoff sends: ", cut.doc_ids, f"({cut.cutoff_k} passage{'s' if cut.cutoff_k != 1 else ''})")
