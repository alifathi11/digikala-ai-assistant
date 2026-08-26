import pandas as pd

from src.rag.product_search.review_evidence import (
    CandidateReviewEvidenceRetriever,
)


class FakeRetriever:

    def __init__(
        self,
        documents,
    ):
        self.documents = documents
        self.calls = []


    def retrieve(
        self,
        query,
        top_k=5,
        candidate_ids=None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "candidate_ids": list(
                    candidate_ids
                ),
            }
        )

        rows = (
            self.documents[
                self.documents[
                    "id"
                ].isin(
                    candidate_ids
                )
            ]
            .head(
                top_k
            )
            .copy()
        )

        rows[
            "score"
        ] = 1.0

        return rows


def test_candidate_review_retrieval_is_product_scoped():
    documents = pd.DataFrame(
        [
            {
                "id": 1,
                "product_id": 10,
                "body": "a",
            },
            {
                "id": 2,
                "product_id": 10,
                "body": "b",
            },
            {
                "id": 3,
                "product_id": 20,
                "body": "c",
            },
            {
                "id": 4,
                "product_id": 30,
                "body": "d",
            },
        ]
    )

    base = FakeRetriever(
        documents
    )

    scoped = (
        CandidateReviewEvidenceRetriever(
            retriever=base,
            reviews_per_product=2,
        )
    )

    evidence, telemetry = (
        scoped.retrieve(
            query="query",
            product_ids=[
                10,
                20,
            ],
        )
    )

    assert len(
        base.calls
    ) == 2

    assert set(
        base.calls[0][
            "candidate_ids"
        ]
    ) == {
        1,
        2,
    }

    assert set(
        base.calls[1][
            "candidate_ids"
        ]
    ) == {
        3,
    }

    assert set(
        evidence[
            "product_id"
        ]
    ) == {
        10,
        20,
    }

    assert (
        telemetry[
            "review_scope"
        ]
        == "candidate_product"
    )

    assert (
        telemetry[
            "retrieved_review_count"
        ]
        == 3
    )
