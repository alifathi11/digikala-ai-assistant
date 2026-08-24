import pandas as pd

from rank_bm25 import BM25Okapi

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        documents: pd.DataFrame,
        text_column: str = "search_text"
    ):

        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )

        self.text_column = text_column

        self._build_index()


    def _build_index(self):

        texts = (
            self.documents[self.text_column]
            .fillna("")
            .astype(str)
        )

        self.tokenized_documents = [
            text.split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        results = self.documents.copy()

        results["score"] = scores

        return (
            results
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )