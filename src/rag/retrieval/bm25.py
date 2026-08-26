from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tantivy

from .base import BaseRetriever
from ..utils.text import build_comment_text


class BM25Retriever(BaseRetriever):
    """
    Disk-backed BM25 retriever powered by Tantivy.

    Public contract:
        retrieve(query, top_k=5, candidate_ids=None)

    `id` in returned rows is always the original comment ID.
    `doc_index` is the row position used by the index/metadata.
    """

    INDEX_SUBDIR = "tantivy"
    METADATA_FILE = "metadata.parquet"
    MANIFEST_FILE = "manifest.json"

    ANALYZER_NAME = "fa_whitespace"
    TEXT_FIELD = "search_text"
    DOC_INDEX_FIELD = "doc_index"
    COMMENT_ID_FIELD = "comment_id"
    PRODUCT_ID_FIELD = "product_id"

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
        self,
        documents=None,
        processor=None,
        text_column="search_text",
    ):
        super().__init__(processor)

        self.documents = None
        self.text_column = text_column
        self.index = None
        self.index_path = None

        if documents is not None:
            self.build(documents)


    @classmethod
    def _schema(cls):
        builder = tantivy.SchemaBuilder()

        builder.add_integer_field(
            cls.DOC_INDEX_FIELD,
            stored=True,
            indexed=True,
        )

        builder.add_integer_field(
            cls.COMMENT_ID_FIELD,
            stored=True,
            indexed=True,
        )

        builder.add_integer_field(
            cls.PRODUCT_ID_FIELD,
            stored=True,
            indexed=True,
        )

        builder.add_text_field(
            cls.TEXT_FIELD,
            stored=False,
            tokenizer_name=cls.ANALYZER_NAME,
        )

        return builder.build()


    @classmethod
    def _analyzer(cls):
        # TextProcessor already handles Persian Arabic-character normalization.
        # Whitespace tokenization intentionally matches the old `text.split()`
        # behavior used by rank_bm25.
        return (
            tantivy.TextAnalyzerBuilder(
                tantivy.Tokenizer.whitespace()
            )
            .build()
        )


    @classmethod
    def _register_analyzer(cls, index):
        index.register_tokenizer(
            cls.ANALYZER_NAME,
            cls._analyzer(),
        )


    @staticmethod
    def _as_int(value, default=-1):
        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass

        return int(value)


    @staticmethod
    def _stored_scalar(document, field):
        if hasattr(document, "get_first"):
            return document.get_first(field)

        value = document[field]

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return value[0]

        return value


    @staticmethod
    def _safe_text(value):
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value)


    @classmethod
    def _prepare_batch(
        cls,
        batch_df,
        processor=None,
    ):
        batch_df = batch_df.copy()

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
            search_text = search_text.map(
                processor.process
            )

        batch_df["search_text"] = search_text

        metadata = (
            batch_df[
                list(cls.METADATA_COLUMNS)
            ]
            .reset_index(drop=True)
            .copy()
        )

        return metadata


    @classmethod
    def build_from_parquet(
        cls,
        input_path,
        output_path,
        processor=None,
        batch_size=50_000,
        writer_heap_size=128_000_000,
        num_threads=1,
        commit_every_batches=10,
        overwrite=False,
    ):
        """
        Build one GLOBAL Tantivy BM25 index without loading the full corpus
        into pandas/RAM.

        Parquet is read batch-by-batch. Tantivy controls indexing memory via
        `writer_heap_size` and writes index segments to disk.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input parquet not found: {input_path}"
            )

        if output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. "
                    "Use overwrite=True to rebuild it."
                )

            shutil.rmtree(output_path)

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        tantivy_path = (
            output_path
            / cls.INDEX_SUBDIR
        )

        tantivy_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = (
            output_path
            / cls.METADATA_FILE
        )

        parquet_file = pq.ParquetFile(
            input_path
        )

        available_columns = set(
            parquet_file.schema.names
        )

        missing = (
            set(cls.SOURCE_COLUMNS)
            - available_columns
        )

        if missing:
            raise ValueError(
                "Missing required comment columns: "
                f"{sorted(missing)}"
            )

        schema = cls._schema()

        index = tantivy.Index(
            schema,
            path=str(tantivy_path),
        )

        cls._register_analyzer(
            index
        )

        writer = index.writer(
            heap_size=int(
                writer_heap_size
            ),
            num_threads=int(
                num_threads
            ),
        )

        metadata_writer = None
        total_documents = 0
        batch_number = 0

        try:
            batches = (
                parquet_file
                .iter_batches(
                    batch_size=int(
                        batch_size
                    ),
                    columns=list(
                        cls.SOURCE_COLUMNS
                    ),
                )
            )

            for arrow_batch in batches:
                batch_number += 1

                batch_df = (
                    arrow_batch
                    .to_pandas()
                )

                metadata = (
                    cls._prepare_batch(
                        batch_df,
                        processor=processor,
                    )
                )

                for local_index, row in enumerate(
                    metadata.itertuples(
                        index=False
                    )
                ):
                    doc_index = (
                        total_documents
                        + local_index
                    )

                    writer.add_document(
                        tantivy.Document(
                            doc_index=int(
                                doc_index
                            ),
                            comment_id=cls._as_int(
                                row.id
                            ),
                            product_id=cls._as_int(
                                row.product_id
                            ),
                            search_text=cls._safe_text(
                                row.search_text
                            ),
                        )
                    )

                table = pa.Table.from_pandas(
                    metadata,
                    preserve_index=False,
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

                if (
                    commit_every_batches
                    and batch_number
                    % int(
                        commit_every_batches
                    )
                    == 0
                ):
                    writer.commit()

                print(
                    f"Indexed "
                    f"{total_documents:,} documents"
                )

                del batch_df
                del metadata
                del table

            writer.commit()

            # Must be called only after the final commit.
            writer.wait_merging_threads()

        finally:
            if metadata_writer is not None:
                metadata_writer.close()

        index.reload()

        manifest = {
            "backend": "tantivy",
            "tantivy_version": getattr(
                tantivy,
                "__version__",
                "unknown",
            ),
            "num_documents": int(
                total_documents
            ),
            "batch_size": int(
                batch_size
            ),
            "writer_heap_size": int(
                writer_heap_size
            ),
            "num_threads": int(
                num_threads
            ),
            "commit_every_batches": int(
                commit_every_batches
            )
            if commit_every_batches
            else None,
            "tokenizer": "whitespace",
            "text_field": cls.TEXT_FIELD,
            "metadata_file": cls.METADATA_FILE,
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


    def build(
        self,
        documents,
        writer_heap_size=64_000_000,
    ):
        """
        Build a small in-memory Tantivy index.

        Intended for tests/small corpora. For the real Digikala corpus use
        `build_from_parquet`.
        """
        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )

        if self.text_column not in self.documents:
            raise ValueError(
                f"Missing text column: "
                f"{self.text_column}"
            )

        schema = self._schema()

        self.index = tantivy.Index(
            schema
        )

        self._register_analyzer(
            self.index
        )

        writer = self.index.writer(
            heap_size=int(
                writer_heap_size
            ),
            num_threads=1,
        )

        for doc_index, row in enumerate(
            self.documents.itertuples(
                index=False
            )
        ):
            row_dict = row._asdict()

            writer.add_document(
                tantivy.Document(
                    doc_index=int(
                        doc_index
                    ),
                    comment_id=self._as_int(
                        row_dict.get("id")
                    ),
                    product_id=self._as_int(
                        row_dict.get(
                            "product_id"
                        )
                    ),
                    search_text=self._safe_text(
                        row_dict.get(
                            self.text_column,
                            ""
                        )
                    ),
                )
            )

        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()


    def load(
        self,
        path,
    ):
        path = Path(path)

        tantivy_path = (
            path
            / self.INDEX_SUBDIR
        )

        metadata_path = (
            path
            / self.METADATA_FILE
        )

        if not tantivy_path.exists():
            raise FileNotFoundError(
                "Tantivy index not found: "
                f"{tantivy_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "BM25 metadata not found: "
                f"{metadata_path}"
            )

        self.index = tantivy.Index.open(
            str(tantivy_path)
        )

        self._register_analyzer(
            self.index
        )

        self.index_path = path

        self.documents = (
            pd.read_parquet(
                metadata_path
            )
            .reset_index(drop=True)
        )

        return self


    def _build_text_query(
        self,
        query,
    ):
        terms = [
            token
            for token in query.split()
            if token
        ]

        if not terms:
            return None

        term_queries = [
            tantivy.Query.term_query(
                self.index.schema,
                self.TEXT_FIELD,
                term,
            )
            for term in terms
        ]

        return tantivy.Query.boolean_query(
            [
                (
                    tantivy.Occur.Should,
                    term_query,
                )
                for term_query in term_queries
            ],
            minimum_number_should_match=1,
        )


    def _apply_candidate_filter(
        self,
        query,
        candidate_ids,
    ):
        if candidate_ids is None:
            return query

        candidate_ids = [
            int(comment_id)
            for comment_id
            in candidate_ids
        ]

        if not candidate_ids:
            return None

        candidate_query = (
            tantivy.Query
            .term_set_query(
                self.index.schema,
                self.COMMENT_ID_FIELD,
                candidate_ids,
            )
        )

        return tantivy.Query.boolean_query(
            [
                (
                    tantivy.Occur.Must,
                    query,
                ),
                (
                    tantivy.Occur.Must,
                    candidate_query,
                ),
            ]
        )


    def _format_hits(
        self,
        searcher,
        hits,
    ):
        if not hits:
            columns = [
                "doc_index",
                "score",
                *self.documents.columns.tolist(),
            ]

            return pd.DataFrame(
                columns=columns
            )

        row_ids = []
        scores = []

        for score, doc_address in hits:
            document = searcher.doc(
                doc_address
            )

            doc_index = self._stored_scalar(
                document,
                self.DOC_INDEX_FIELD,
            )

            row_ids.append(
                int(doc_index)
            )

            scores.append(
                float(score)
            )

        row_ids = np.asarray(
            row_ids,
            dtype=np.int64,
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
            np.asarray(
                scores,
                dtype=float,
            ),
        )

        return results


    def retrieve(
        self,
        query,
        top_k=5,
        candidate_ids=None,
    ):
        if self.index is None:
            raise RuntimeError(
                "BM25 index is not loaded."
            )

        query = self.process_query(
            query
        )

        text_query = (
            self._build_text_query(
                query
            )
        )

        if text_query is None:
            return self._format_hits(
                self.index.searcher(),
                [],
            )

        final_query = (
            self._apply_candidate_filter(
                text_query,
                candidate_ids,
            )
        )

        if final_query is None:
            return self._format_hits(
                self.index.searcher(),
                [],
            )

        k = int(top_k)

        if candidate_ids is not None:
            k = min(
                k,
                len(candidate_ids),
            )

        if k <= 0:
            return self._format_hits(
                self.index.searcher(),
                [],
            )

        searcher = self.index.searcher()

        search_result = searcher.search(
            final_query,
            limit=k,
            count=False,
        )

        return self._format_hits(
            searcher,
            search_result.hits,
        )
