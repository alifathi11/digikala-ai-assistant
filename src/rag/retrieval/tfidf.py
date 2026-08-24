import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever


class TFIDFRetriever(BaseRetriever):

    def __init__(
        self,
        documents: pd.DataFrame,
        text_column: str = "search_text"
    ):

        self.documents = documents.reset_index(drop=True)

        self.text_column = text_column

        self.vectorizer = TfidfVectorizer()

        self._build_index()


    def _build_index(self):

        valid_docs = (
            self.documents[self.text_column]
            .fillna("")
            .str.strip()
        )

        mask = valid_docs.str.len() > 0

        self.documents = (
            self.documents[mask]
            .reset_index(drop=True)
        )

        self.document_vectors = (
            self.vectorizer.fit_transform(
                self.documents[self.text_column]
            )
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        query_vector = (
            self.vectorizer
            .transform([query])
        )

        scores = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]


        result = self.documents.copy()

        result["score"] = scores


        return (
            result
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )