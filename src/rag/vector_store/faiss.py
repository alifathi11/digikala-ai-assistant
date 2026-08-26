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
        self._comment_id_to_row = None


    @staticmethod
    def _chunk_start(file):
        return int(
            Path(file)
            .stem
            .split("_")[-1]
        )


    def load(self, path):
        path = Path(path)

        self.chunk_files = sorted(
            glob.glob(
                str(path / "chunks" / "*.faiss")
            ),
            key=self._chunk_start
        )

        self.offsets = [
            self._chunk_start(file)
            for file in self.chunk_files
        ]

        self.documents = (
            pd.read_parquet(
                path / "metadata.parquet"
            )
            .reset_index(drop=True)
        )

        self._comment_id_to_row = {
            int(comment_id): int(row_idx)
            for row_idx, comment_id in enumerate(
                self.documents["id"].tolist()
            )
        }


    def _format_results(
        self,
        row_ids,
        scores
    ):
        if len(row_ids) == 0:
            columns = [
                "doc_index",
                "score",
                *self.documents.columns.tolist()
            ]
            return pd.DataFrame(columns=columns)

        row_ids = np.asarray(
            row_ids,
            dtype=np.int64
        )

        scores = np.asarray(
            scores,
            dtype=float
        )

        results = (
            self.documents
            .iloc[row_ids]
            .copy()
            .reset_index(drop=True)
        )

        results.insert(
            0,
            "doc_index",
            row_ids
        )

        results.insert(
            1,
            "score",
            scores
        )

        return results


    def _candidate_rows(
        self,
        candidate_ids
    ):
        if candidate_ids is None:
            return None

        missing = [
            int(comment_id)
            for comment_id in candidate_ids
            if int(comment_id)
            not in self._comment_id_to_row
        ]

        if missing:
            raise ValueError(
                "Some candidate IDs are missing from FAISS metadata: "
                f"{missing[:5]}"
            )

        return np.asarray(
            [
                self._comment_id_to_row[
                    int(comment_id)
                ]
                for comment_id in candidate_ids
            ],
            dtype=np.int64
        )


    def _search_candidates(
        self,
        query_embedding,
        candidate_ids,
        top_k
    ):
        candidate_rows = self._candidate_rows(
            candidate_ids
        )

        if len(candidate_rows) == 0:
            return self._format_results([], [])

        vectors = []
        rows = []

        offsets = np.asarray(
            self.offsets,
            dtype=np.int64
        )

        chunk_ids = (
            np.searchsorted(
                offsets,
                candidate_rows,
                side="right"
            )
            - 1
        )

        for chunk_id in np.unique(chunk_ids):
            selected_rows = candidate_rows[
                chunk_ids == chunk_id
            ]

            local_ids = (
                selected_rows
                - offsets[chunk_id]
            )

            index = faiss.read_index(
                self.chunk_files[chunk_id]
            )

            chunk_vectors = np.vstack(
                [
                    index.reconstruct(
                        int(local_id)
                    )
                    for local_id in local_ids
                ]
            ).astype("float32")

            vectors.append(chunk_vectors)
            rows.append(selected_rows)

            del index

        vectors = np.vstack(vectors)
        rows = np.concatenate(rows)

        query_vector = query_embedding[0]

        scores = vectors @ query_vector

        k = min(top_k, len(scores))
        order = np.argsort(-scores)[:k]

        return self._format_results(
            rows[order],
            scores[order]
        )


    def search(
        self,
        query_embedding,
        top_k=5,
        candidate_ids=None
    ):
        query_embedding = np.asarray(
            query_embedding
        ).astype("float32")

        if candidate_ids is not None:
            return self._search_candidates(
                query_embedding,
                candidate_ids,
                top_k
            )

        all_results = []

        for file, offset in zip(
            self.chunk_files,
            self.offsets
        ):
            index = faiss.read_index(file)

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
                        "doc_index": int(
                            idx + offset
                        ),
                        "score": float(score)
                    }
                )

            del index

        if not all_results:
            return self._format_results([], [])

        results = (
            pd.DataFrame(all_results)
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )

        return self._format_results(
            results["doc_index"].to_numpy(),
            results["score"].to_numpy()
        )
