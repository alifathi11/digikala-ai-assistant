from pathlib import Path
import glob
import json
import shutil

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.text import build_comment_text


class FAISSVectorStore:
    """
    Chunked FAISS vector store.

    The production index is disk-backed as independent FAISS chunks so index
    construction and retrieval stay within a bounded memory footprint.
    """

    CHUNKS_DIR = "chunks"
    METADATA_FILE = "metadata.parquet"
    MANIFEST_FILE = "manifest.json"

    SOURCE_COLUMNS = (
        "id",
        "product_id",
        "title",
        "body",
        "advantages",
        "disadvantages",
        "rate",
    )

    METADATA_COLUMNS = (
        "id",
        "product_id",
        "body",
        "rate",
        "search_text",
    )


    def __init__(
        self
    ):
        self.chunk_files = []
        self.offsets = []
        self.documents = None
        self._comment_id_to_row = None


    @staticmethod
    def _chunk_start(
        file
    ):
        return int(
            Path(file)
            .stem
            .split("_")[-1]
        )


    @classmethod
    def build_from_parquet(
        cls,
        input_path,
        output_path,
        embedding_model,
        processor=None,
        chunk_size=5_000,
        encode_batch_size=64,
        device=None,
        overwrite=False,
    ):
        """
        Build the chunked FAISS index without loading the full comments corpus
        into pandas or materializing every embedding in RAM.
        """
        input_path = Path(
            input_path
        )

        output_path = Path(
            output_path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input parquet not found: "
                f"{input_path}"
            )

        if output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. "
                    "Use overwrite=True to rebuild it."
                )

            shutil.rmtree(
                output_path
            )

        chunk_dir = (
            output_path
            / cls.CHUNKS_DIR
        )

        chunk_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = (
            output_path
            / cls.METADATA_FILE
        )

        parquet_file = (
            pq.ParquetFile(
                input_path
            )
        )

        available = set(
            parquet_file
            .schema
            .names
        )

        missing = (
            set(
                cls.SOURCE_COLUMNS
            )
            - available
        )

        if missing:
            raise ValueError(
                "Missing required comment "
                f"columns: {sorted(missing)}"
            )

        metadata_writer = None
        total_documents = 0
        vector_dimension = None

        try:
            for arrow_batch in (
                parquet_file
                .iter_batches(
                    batch_size=int(
                        chunk_size
                    ),
                    columns=list(
                        cls.SOURCE_COLUMNS
                    ),
                )
            ):
                batch_df = (
                    arrow_batch
                    .to_pandas()
                )

                search_text = (
                    batch_df
                    .apply(
                        build_comment_text,
                        axis=1,
                    )
                    .fillna("")
                    .astype(str)
                )

                if processor is not None:
                    search_text = (
                        search_text
                        .map(
                            processor.process
                        )
                    )

                metadata = (
                    batch_df[
                        [
                            "id",
                            "product_id",
                            "body",
                            "rate",
                        ]
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

                metadata[
                    "search_text"
                ] = (
                    search_text
                    .reset_index(drop=True)
                )

                texts = (
                    metadata[
                        "search_text"
                    ]
                    .tolist()
                )

                encode_kwargs = {
                    "batch_size": int(
                        encode_batch_size
                    ),
                }

                if device is not None:
                    encode_kwargs[
                        "device"
                    ] = device

                embeddings = (
                    embedding_model
                    .encode(
                        texts,
                        **encode_kwargs,
                    )
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype="float32",
                )

                if embeddings.ndim != 2:
                    raise ValueError(
                        "Embedding model must "
                        "return a 2D matrix"
                    )

                faiss.normalize_L2(
                    embeddings
                )

                vector_dimension = int(
                    embeddings.shape[1]
                )

                index = faiss.IndexFlatIP(
                    vector_dimension
                )

                index.add(
                    embeddings
                )

                chunk_path = (
                    chunk_dir
                    / (
                        f"chunk_"
                        f"{total_documents}"
                        f".faiss"
                    )
                )

                faiss.write_index(
                    index,
                    str(
                        chunk_path
                    ),
                )

                table = (
                    pa.Table
                    .from_pandas(
                        metadata[
                            list(
                                cls.METADATA_COLUMNS
                            )
                        ],
                        preserve_index=False,
                    )
                )

                if metadata_writer is None:
                    metadata_writer = (
                        pq.ParquetWriter(
                            metadata_path,
                            table.schema,
                            compression="zstd",
                        )
                    )

                metadata_writer.write_table(
                    table
                )

                total_documents += len(
                    metadata
                )

                print(
                    "Indexed "
                    f"{total_documents:,} "
                    "documents"
                )

                del embeddings
                del index
                del batch_df
                del metadata
                del table

        finally:
            if metadata_writer is not None:
                metadata_writer.close()

        manifest = {
            "backend": "faiss",
            "index_type": "IndexFlatIP",
            "num_documents": int(
                total_documents
            ),
            "chunk_size": int(
                chunk_size
            ),
            "vector_dimension": (
                vector_dimension
            ),
            "normalized_embeddings": True,
            "metadata_file": (
                cls.METADATA_FILE
            ),
        }

        with open(
            output_path
            / cls.MANIFEST_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return manifest


    def load(
        self,
        path,
    ):
        path = Path(
            path
        )

        self.chunk_files = sorted(
            glob.glob(
                str(
                    path
                    / self.CHUNKS_DIR
                    / "*.faiss"
                )
            ),
            key=self._chunk_start,
        )

        if not self.chunk_files:
            raise FileNotFoundError(
                "No FAISS chunks found in "
                f"{path / self.CHUNKS_DIR}"
            )

        self.offsets = [
            self._chunk_start(
                file
            )
            for file in self.chunk_files
        ]

        metadata_path = (
            path
            / self.METADATA_FILE
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS metadata not found: "
                f"{metadata_path}"
            )

        self.documents = (
            pd.read_parquet(
                metadata_path
            )
            .reset_index(drop=True)
        )

        self._comment_id_to_row = {
            int(comment_id): int(
                row_idx
            )
            for row_idx, comment_id
            in enumerate(
                self.documents[
                    "id"
                ].tolist()
            )
        }

        return self


    def _format_results(
        self,
        row_ids,
        scores,
    ):
        if len(row_ids) == 0:
            columns = [
                "doc_index",
                "score",
                *self.documents.columns.tolist(),
            ]

            return pd.DataFrame(
                columns=columns
            )

        row_ids = np.asarray(
            row_ids,
            dtype=np.int64,
        )

        scores = np.asarray(
            scores,
            dtype=float,
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
            row_ids,
        )

        results.insert(
            1,
            "score",
            scores,
        )

        return results


    def _candidate_rows(
        self,
        candidate_ids,
    ):
        if candidate_ids is None:
            return None

        missing = [
            int(comment_id)
            for comment_id
            in candidate_ids
            if int(comment_id)
            not in self._comment_id_to_row
        ]

        if missing:
            raise ValueError(
                "Some candidate IDs are "
                "missing from FAISS metadata: "
                f"{missing[:5]}"
            )

        return np.asarray(
            [
                self._comment_id_to_row[
                    int(comment_id)
                ]
                for comment_id
                in candidate_ids
            ],
            dtype=np.int64,
        )


    def _search_candidates(
        self,
        query_embedding,
        candidate_ids,
        top_k,
    ):
        candidate_rows = (
            self._candidate_rows(
                candidate_ids
            )
        )

        if len(candidate_rows) == 0:
            return self._format_results(
                [],
                [],
            )

        vectors = []
        rows = []

        offsets = np.asarray(
            self.offsets,
            dtype=np.int64,
        )

        chunk_ids = (
            np.searchsorted(
                offsets,
                candidate_rows,
                side="right",
            )
            - 1
        )

        for chunk_id in np.unique(
            chunk_ids
        ):
            selected_rows = (
                candidate_rows[
                    chunk_ids
                    == chunk_id
                ]
            )

            local_ids = (
                selected_rows
                - offsets[
                    chunk_id
                ]
            )

            index = faiss.read_index(
                self.chunk_files[
                    chunk_id
                ]
            )

            chunk_vectors = np.vstack(
                [
                    index.reconstruct(
                        int(
                            local_id
                        )
                    )
                    for local_id
                    in local_ids
                ]
            ).astype(
                "float32"
            )

            vectors.append(
                chunk_vectors
            )

            rows.append(
                selected_rows
            )

            del index

        vectors = np.vstack(
            vectors
        )

        rows = np.concatenate(
            rows
        )

        query_vector = (
            query_embedding[0]
        )

        scores = (
            vectors
            @ query_vector
        )

        k = min(
            int(top_k),
            len(scores),
        )

        order = np.argsort(
            -scores
        )[:k]

        return self._format_results(
            rows[order],
            scores[order],
        )


    def search(
        self,
        query_embedding,
        top_k=5,
        candidate_ids=None,
    ):
        if self.documents is None:
            raise RuntimeError(
                "FAISS vector store is "
                "not loaded."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        faiss.normalize_L2(
            query_embedding
        )

        if candidate_ids is not None:
            return (
                self._search_candidates(
                    query_embedding,
                    candidate_ids,
                    top_k,
                )
            )

        all_results = []

        for file, offset in zip(
            self.chunk_files,
            self.offsets,
        ):
            index = faiss.read_index(
                file
            )

            scores, ids = (
                index.search(
                    query_embedding,
                    int(top_k),
                )
            )

            for score, idx in zip(
                scores[0],
                ids[0],
            ):
                if idx == -1:
                    continue

                all_results.append(
                    {
                        "doc_index": int(
                            idx
                            + offset
                        ),
                        "score": float(
                            score
                        ),
                    }
                )

            del index

        if not all_results:
            return self._format_results(
                [],
                [],
            )

        results = (
            pd.DataFrame(
                all_results
            )
            .sort_values(
                "score",
                ascending=False,
            )
            .head(
                int(top_k)
            )
        )

        return self._format_results(
            results[
                "doc_index"
            ].to_numpy(),
            results[
                "score"
            ].to_numpy(),
        )
