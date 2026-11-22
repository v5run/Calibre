from calibre_rag.eval import ndcg_at_k, recall_at_k, selection_recall


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}) == 1.0
    assert ndcg_at_k(["x"], {"a": 1}) == 0.0


def test_recall_variants():
    assert recall_at_k(["a", "x"], {"a": 1, "b": 1}, 2) == 0.5
    assert selection_recall(["a"], {"a": 1}) == 1.0
