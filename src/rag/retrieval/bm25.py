from pathlib import Path
import glob
import pickle

import pandas as pd

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        documents=None,
        processor=None,
        text_column="search_text"
    ):

        super().__init__(
            processor
        )

        self.documents = None
        self.text_column = text_column

        self.chunk_files = []
        self.offsets = []


        if documents is not None:
            self.build(
                documents
            )


    def build(
        self,
        documents
    ):

        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )


    def load(self, path):

        path = Path(path)

        self.chunk_files = glob.glob(
            str(path / "chunks" / "*.pkl")
        )

        self.chunk_files = sorted(
            self.chunk_files,
            key=self._get_chunk_start
        )

        self.offsets = [
            self._get_chunk_start(file)
            for file in self.chunk_files
        ]

        self.documents = pd.read_parquet(
            path / "metadata.parquet"
        )


    def retrieve(
        self,
        query,
        top_k=5
    ):

        query = self.process_query(
            query
        )


        tokenized_query = query.split()


        all_results = []


        for file, offset in zip(
            self.chunk_files,
            self.offsets
        ):

            with open(
                file,
                "rb"
            ) as f:

                bm25 = pickle.load(
                    f
                )


            scores = bm25.get_scores(
                tokenized_query
            )


            for idx, score in enumerate(scores):

                all_results.append(
                    {
                        "id": idx + offset,
                        "score": float(score)
                    }
                )


            del bm25



        results = (
            pd.DataFrame(
                all_results
            )
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )


        results = results.merge(
            self.documents,
            left_on="id",
            right_index=True,
            how="left"
        )


        return results

    def _get_chunk_start(self, file):
        return int(
            Path(file).stem.split("_")[-1]
        )