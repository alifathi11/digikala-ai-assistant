from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Return top-k relevant documents.
        """
        pass