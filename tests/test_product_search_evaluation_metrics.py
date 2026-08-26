import pandas as pd

from src.rag.evaluation.product_search_metrics import rank_policy


def test_llm_tiered_demotes_metadata_false_positive():
    candidates = pd.DataFrame([
        {"id": 1, "metadata_rank": 1, "metadata_score": 0.95, "llm_match_score": 0.0},
        {"id": 2, "metadata_rank": 2, "metadata_score": 0.75, "llm_match_score": 5.0},
    ])
    policy = {
        "name": "tiered", "kind": "llm_tiered",
        "metadata_weight": 0.3, "llm_weight": 0.7, "reranker_k": 12,
    }
    ranked = rank_policy(candidates, policy)
    assert int(ranked.iloc[0]["id"]) == 2
