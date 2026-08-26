
import pandas as pd

from src.rag.pipeline.product_search import (
    ProductSearchPipeline,
)


class FakeMetadataRetriever:

    def __init__(self):
        self.documents = pd.DataFrame(
            [
                {"id": 1},
                {"id": 2},
            ]
        )

    def retrieve(
        self,
        query,
        top_k=50,
    ):
        return pd.DataFrame(
            [
                {
                    "id": 1,
                    "title_fa": "برس مو",
                    "metadata_score": 1.0,
                },
                {
                    "id": 2,
                    "title_fa": "ضد آفتاب پوست چرب",
                    "metadata_score": 0.1,
                },
            ]
        )


class FakeReviewRetriever:

    def __init__(self):
        self.documents = pd.DataFrame(
            columns=[
                "id",
                "product_id",
            ]
        )


class FakeReranker:

    def rerank(
        self,
        query,
        candidates,
        review_comments,
    ):
        return (
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "llm_match_score": 0.0,
                        "evidence_status": "none",
                        "evidence_ids": [],
                        "reason": "irrelevant",
                    },
                    {
                        "id": 2,
                        "llm_match_score": 5.0,
                        "evidence_status": "none",
                        "evidence_ids": [],
                        "reason": "exact type match",
                    },
                ]
            ),
            {},
        )


def test_llm_reranker_dominates_bad_metadata_order():
    pipeline = ProductSearchPipeline(
        metadata_retriever=FakeMetadataRetriever(),
        review_retriever=FakeReviewRetriever(),
        reranker=FakeReranker(),
        metadata_candidates=2,
        reranker_candidates=2,
        metadata_weight=0.30,
        reranker_weight=0.70,
        review_comments_per_product=1,
    )

    results = pipeline.search(
        "ضد آفتاب پوست چرب",
        top_k=2,
    )

    assert int(
        results.iloc[0]["id"]
    ) == 2

    assert float(
        results.iloc[0][
            "llm_match_score"
        ]
    ) == 5.0

    assert int(
        results.iloc[-1]["id"]
    ) == 1
