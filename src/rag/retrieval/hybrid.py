import pandas as pd

from .base import BaseRetriever


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        bm25_retriever,
        embedding_retriever,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7
    ):

        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever

        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight


    def _normalize(self, scores):

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return scores * 0

        return (
            scores - min_score
        ) / (
            max_score - min_score
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        candidate_k = top_k * 3


        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                candidate_k
            )
        )


        embedding_results = (
            self.embedding_retriever.retrieve(
                query,
                candidate_k
            )
        )


        bm25_results = bm25_results.copy()
        embedding_results = embedding_results.copy()


        bm25_results["bm25_score"] = (
            self._normalize(
                bm25_results["score"]
            )
        )


        embedding_results["embedding_score"] = (
            self._normalize(
                embedding_results["score"]
            )
        )


        merged = pd.concat(
            [
                bm25_results[
                    [
                        "body",
                        "bm25_score"
                    ]
                ],
                embedding_results[
                    [
                        "body",
                        "embedding_score"
                    ]
                ]
            ]
        )


        merged = (
            merged
            .groupby("body", dropna=False)
            .max()
            .reset_index()
        )


        merged["score"] = (
            self.bm25_weight *
            merged.get(
                "bm25_score",
                0
            )
            +
            self.embedding_weight *
            merged.get(
                "embedding_score",
                0
            )
        )


        return (
            merged
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )