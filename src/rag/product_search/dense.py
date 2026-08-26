from pathlib import Path
import glob
import json
import shutil

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ProductFAISSIndex:

    CHUNKS_DIR = "chunks"
    METADATA_FILE = "metadata.parquet"
    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
    ):
        self.chunk_files = []
        self.offsets = []
        self.documents = None
        self._product_id_to_row = None


    @staticmethod
    def _chunk_start(
        file,
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
        chunk_size=10_000,
        encode_batch_size=128,
        device=None,
        overwrite=False,
    ):
        input_path = Path(
            input_path
        )

        output_path = Path(
            output_path
        )

        if output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_path} exists. "
                    "Use overwrite=True."
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

        required = {
            "id",
            "search_text",
        }

        missing = (
            required
            - set(
                parquet_file
                .schema
                .names
            )
        )

        if missing:
            raise ValueError(
                "Canonical products missing: "
                f"{sorted(missing)}"
            )

        metadata_columns = [
            column
            for column
            in parquet_file.schema.names
            if column
            not in {
                "Seller",
            }
        ]

        metadata_writer = None
        total_documents = 0
        vector_dimension = None

        try:
            for batch in (
                parquet_file
                .iter_batches(
                    batch_size=int(
                        chunk_size
                    ),
                    columns=(
                        metadata_columns
                    ),
                )
            ):
                frame = (
                    batch
                    .to_pandas()
                    .reset_index(drop=True)
                )

                texts = (
                    frame[
                        "search_text"
                    ]
                    .fillna("")
                    .astype(str)
                )

                if processor is not None:
                    texts = (
                        texts
                        .map(
                            processor.process
                        )
                    )

                frame[
                    "search_text"
                ] = texts

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
                        texts.tolist(),
                        **encode_kwargs,
                    )
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype="float32",
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

                faiss.write_index(
                    index,
                    str(
                        chunk_dir
                        / (
                            "chunk_"
                            f"{total_documents}"
                            ".faiss"
                        )
                    ),
                )

                table = (
                    pa.Table
                    .from_pandas(
                        frame,
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
                    frame
                )

                print(
                    "Indexed "
                    f"{total_documents:,} "
                    "products"
                )

                del embeddings
                del index
                del frame
                del table

        finally:
            if metadata_writer is not None:
                metadata_writer.close()

        manifest = {
            "backend": "faiss",
            "num_documents": int(
                total_documents
            ),
            "dimension": int(
                vector_dimension
            ),
            "chunk_size": int(
                chunk_size
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
                "No product FAISS chunks found"
            )

        self.offsets = [
            self._chunk_start(
                file
            )
            for file in self.chunk_files
        ]

        self.documents = (
            pd.read_parquet(
                path
                / self.METADATA_FILE
            )
            .reset_index(drop=True)
        )

        self._product_id_to_row = {
            int(product_id): int(row_idx)
            for row_idx, product_id
            in enumerate(
                self.documents[
                    "id"
                ].tolist()
            )
        }

        return self


    def _format(
        self,
        row_ids,
        scores,
    ):
        results = (
            self.documents
            .iloc[
                np.asarray(
                    row_ids,
                    dtype=np.int64,
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        results.insert(
            0,
            "doc_index",
            np.asarray(
                row_ids,
                dtype=np.int64,
            ),
        )

        results.insert(
            1,
            "score",
            np.asarray(
                scores,
                dtype=float,
            ),
        )

        return results


    def search(
        self,
        query_embedding,
        top_k=10,
    ):
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_embedding.ndim == 1:
            query_embedding = (
                query_embedding[None, :]
            )

        faiss.normalize_L2(
            query_embedding
        )

        candidates = []

        for file, offset in zip(
            self.chunk_files,
            self.offsets,
        ):
            index = faiss.read_index(
                file
            )

            k = min(
                int(top_k),
                index.ntotal,
            )

            scores, local_ids = (
                index.search(
                    query_embedding,
                    k,
                )
            )

            for score, local_id in zip(
                scores[0],
                local_ids[0],
            ):
                if local_id < 0:
                    continue

                candidates.append(
                    (
                        int(
                            offset
                            + local_id
                        ),
                        float(score),
                    )
                )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        candidates = candidates[
            :int(top_k)
        ]

        return self._format(
            [
                row_id
                for row_id, _
                in candidates
            ],
            [
                score
                for _, score
                in candidates
            ],
        )
