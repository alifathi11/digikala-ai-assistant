import pandas as pd
import pytest

from src.rag.product_search.retriever import (
    ProductMetadataRetriever,
)


class FakeEmbeddingModel:

    def encode(
        self,
        texts,
    ):
        return [
            [
                1.0,
                0.0,
            ]
            for _ in texts
        ]


class FakeDenseIndex:

    def __init__(
        self,
        documents,
    ):
        self.documents = (
            documents
            .copy()
            .reset_index(drop=True)
        )

        self._product_id_to_row = {
            int(product_id): index
            for index, product_id
            in enumerate(
                self.documents[
                    "id"
                ]
            )
        }


    def search(
        self,
        query_embedding,
        top_k=10,
    ):
        # Deliberate semantic false positive:
        # hair brush is dense rank 1.
        order = [
            1,
            2,
        ]

        scores = {
            1: 0.91,
            2: 0.72,
        }

        frame = (
            self.documents[
                self.documents[
                    "id"
                ].isin(
                    order
                )
            ]
            .set_index(
                "id"
            )
            .loc[
                order
            ]
            .reset_index()
        )

        frame.insert(
            1,
            "score",
            [
                scores[
                    product_id
                ]
                for product_id
                in order
            ],
        )

        return frame.head(
            top_k
        )


class FakeSparseIndex:

    def __init__(
        self,
        documents,
    ):
        self.documents = (
            documents
            .copy()
            .reset_index(drop=True)
        )


    def retrieve(
        self,
        query,
        top_k=10,
    ):
        # Only the actual sunscreen has strong lexical BM25 evidence.
        frame = (
            self.documents[
                self.documents[
                    "id"
                ]
                == 2
            ]
            .copy()
        )

        frame.insert(
            1,
            "score",
            8.0,
        )

        return frame.head(
            top_k
        )


def _documents():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title_fa": (
                    "برس مو کراون مدل آنتی دندروف"
                ),
                "Brand": "کراون",
                "Category1": "زیبایی",
                "Category2": "برس مو",
                "sub_category": "برس مو",
            },
            {
                "id": 2,
                "title_fa": (
                    "کرم ضد آفتاب SPF50 "
                    "مناسب پوست چرب"
                ),
                "Brand": "ژیناژن",
                "Category1": "زیبایی",
                "Category2": "کرم ضد آفتاب",
                "sub_category": "ضد آفتاب",
            },
        ]
    )


def test_lexical_grounding_beats_dense_false_positive():
    documents = _documents()

    retriever = ProductMetadataRetriever(
        embedding_model=(
            FakeEmbeddingModel()
        ),
        dense_index=(
            FakeDenseIndex(
                documents
            )
        ),
        sparse_index=(
            FakeSparseIndex(
                documents
            )
        ),
        bm25_weight=0.5,
        embedding_weight=0.5,
        lexical_weight=0.3,
        brand_boost=0.2,
        rrf_k=60,
    )

    result = retriever.retrieve(
        "ضد آفتاب پوست چرب",
        top_k=2,
    )

    assert int(
        result.iloc[
            0
        ][
            "id"
        ]
    ) == 2

    brush = result[
        result[
            "id"
        ]
        == 1
    ].iloc[0]

    sunscreen = result[
        result[
            "id"
        ]
        == 2
    ].iloc[0]

    assert (
        sunscreen[
            "lexical_score"
        ]
        > brush[
            "lexical_score"
        ]
    )

    assert (
        sunscreen[
            "metadata_score"
        ]
        > brush[
            "metadata_score"
        ]
    )

    # A dense-only false-positive should not become an artificial 100%.
    assert (
        brush[
            "metadata_score"
        ]
        < 0.5
    )


def test_misaligned_product_indexes_fail_fast():
    dense_documents = _documents()

    sparse_documents = (
        _documents()
        .iloc[
            ::-1
        ]
        .reset_index(drop=True)
    )

    with pytest.raises(
        ValueError,
        match="not aligned",
    ):
        ProductMetadataRetriever(
            embedding_model=(
                FakeEmbeddingModel()
            ),
            dense_index=(
                FakeDenseIndex(
                    dense_documents
                )
            ),
            sparse_index=(
                FakeSparseIndex(
                    sparse_documents
                )
            ),
            validate_index_alignment=True,
        )
