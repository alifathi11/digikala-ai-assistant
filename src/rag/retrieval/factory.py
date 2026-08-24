from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever


class RetrieverFactory:

    @staticmethod
    def create(
        retriever_type: str,
        documents,
        **kwargs
    ):

        retriever_type = retriever_type.lower()


        if retriever_type == "tfidf":

            return TFIDFRetriever(
                documents,
                **kwargs
            )


        elif retriever_type == "bm25":

            return BM25Retriever(
                documents,
                **kwargs
            )


        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}"
            )