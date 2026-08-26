
import pandas as pd

from src.rag.product_search.reranker import (
    ProductSearchReranker,
)


def test_support_without_valid_review_is_downgraded_to_none():
    payload = {
        "rankings": [
            {
                "product_id": 10,
                "match_score": 4,
                "evidence_status": "support",
                "evidence_ids": [],
                "reason": "metadata fit",
            }
        ]
    }

    cleaned = ProductSearchReranker._validate(
        payload=payload,
        candidate_ids=[10],
        review_map={
            10: [],
        },
    )

    assert (
        cleaned.iloc[0][
            "evidence_status"
        ]
        == "none"
    )
