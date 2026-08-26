from pathlib import Path
import glob

import faiss
import pandas as pd
import numpy as np


class FAISSVectorStore:

    def __init__(self):

        self.chunk_files = []
        self.offsets = []
        self.documents = None

    def load(self, path):

        path = Path(path)

        self.chunk_files = []
        self.offsets = []

        chunk_files = glob.glob(
            str(path / "chunks" / "*.faiss")
        )

        chunk_files = sorted(
            chunk_files,
            key=self._get_chunk_start
        )

        for file in chunk_files:

            index = faiss.read_index(file)

            chunk_start = self._get_chunk_start(file)

            self.chunk_files.append(file)
            self.offsets.append(chunk_start)

            del index

        self.documents = pd.read_parquet(
            path / "metadata.parquet"
        )

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.asarray(
            query_embedding
        ).astype("float32")


        all_results = []


        for file, offset in zip(
            self.chunk_files,
            self.offsets
        ):

            index = faiss.read_index(
                file
            )


            scores, ids = index.search(
                query_embedding,
                top_k
            )


            for score, idx in zip(
                scores[0],
                ids[0]
            ):

                if idx == -1:
                    continue


                all_results.append(
                    {
                        "id": int(idx + offset),
                        "score": float(score)
                    }
                )


            del index


        results = (
            pd.DataFrame(all_results)
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