from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    def __init__(
        self,
        processor=None
    ):
        self.processor = processor


    def process_query(
        self,
        query
    ):

        if self.processor:
            return self.processor.process(query)

        return query


    @abstractmethod
    def retrieve(
        self,
        query,
        top_k=5
    ):
        pass