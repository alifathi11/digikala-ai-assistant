import numpy as np
import pandas as pd

from .base import BaseRetriever


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        bm25_retriever,
        embedding_retriever,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7,
        candidate_multiplier: int = 3
    ):
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.candidate_multiplier = candidate_multiplier

        vector_store = getattr(
            self.embedding_retriever,
            "vector_store",
            None,
        )

        self._comment_id_to_row = getattr(
            vector_store,
            "_comment_id_to_row",
            None,
        )

        if self._comment_id_to_row is None:
            # Fallback for another vector-store implementation.
            self._comment_id_to_row = {
                int(comment_id): int(row_idx)
                for row_idx, comment_id in enumerate(
                    self.embedding_retriever
                    .documents["id"]
                    .tolist()
                )
            }


    @staticmethod
    def _normalize(scores):
        scores = scores.astype(float)

        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return pd.Series(
                np.zeros(len(scores)),
                index=scores.index
            )

        return (
            scores - min_score
        ) / (
            max_score - min_score
        )


    def _attach_metadata(
        self,
        ranked_results
    ):
        """
        Attach metadata only for final top-k rows.

        The previous implementation called reset_index() on the full
        multi-million-row metadata DataFrame for every query, which
        dominated Hybrid latency.
        """
        if len(ranked_results) == 0:
            return ranked_results

        row_ids = np.asarray(
            [
                self._comment_id_to_row[
                    int(comment_id)
                ]
                for comment_id
                in ranked_results["id"]
            ],
            dtype=np.int64,
        )

        metadata = (
            self.embedding_retriever
            .documents
            .iloc[row_ids]
            .copy()
            .reset_index(drop=True)
        )

        metadata.insert(
            0,
            "doc_index",
            row_ids,
        )

        scores = (
            ranked_results[
                [
                    "score",
                    "bm25_score",
                    "embedding_score",
                ]
            ]
            .reset_index(drop=True)
        )

        return pd.concat(
            [
                metadata,
                scores,
            ],
            axis=1,
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_ids=None
    ):
        candidate_k = max(
            top_k,
            top_k * self.candidate_multiplier
        )

        if candidate_ids is not None:
            candidate_k = min(
                candidate_k,
                len(candidate_ids)
            )

        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                top_k=candidate_k,
                candidate_ids=candidate_ids
            )
        )

        embedding_results = (
            self.embedding_retriever.retrieve(
                query,
                top_k=candidate_k,
                candidate_ids=candidate_ids
            )
        )

        bm25_results = bm25_results[
            [
                "id",
                "score",
            ]
        ].copy()

        embedding_results = embedding_results[
            [
                "id",
                "score",
            ]
        ].copy()

        bm25_results["bm25_score"] = (
            self._normalize(
                bm25_results["score"]
            )
        )

        embedding_results[
            "embedding_score"
        ] = self._normalize(
            embedding_results["score"]
        )

        merged = (
            bm25_results[
                [
                    "id",
                    "bm25_score",
                ]
            ]
            .merge(
                embedding_results[
                    [
                        "id",
                        "embedding_score",
                    ]
                ],
                on="id",
                how="outer",
            )
        )

        merged[
            [
                "bm25_score",
                "embedding_score",
            ]
        ] = merged[
            [
                "bm25_score",
                "embedding_score",
            ]
        ].fillna(0.0)

        merged["score"] = (
            self.bm25_weight
            * merged["bm25_score"]
            +
            self.embedding_weight
            * merged["embedding_score"]
        )

        ranked = (
            merged
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        return self._attach_metadata(
            ranked
        )
