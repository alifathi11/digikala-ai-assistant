from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def encode(
        self,
        texts: list[str]
    ):
        """
        Convert texts into vectors.
        """
        pass