import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever
from ..embedding.base import BaseEmbedding


class EmbeddingRetriever(BaseRetriever):

    def __init__(
        self,
        documents: pd.DataFrame,
        embedding_model: BaseEmbedding,
        text_column: str = "search_text"
    ):

        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )

        self.embedding_model = embedding_model
        self.text_column = text_column

        self._build_index()


    def _build_index(self):

        texts = (
            self.documents[self.text_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.document_embeddings = (
            self.embedding_model.encode(texts)
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )


        scores = cosine_similarity(
            query_embedding,
            self.document_embeddings
        )[0]


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