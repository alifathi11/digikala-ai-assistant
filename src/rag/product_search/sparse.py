from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tantivy


class ProductBM25Index:

    INDEX_SUBDIR = "tantivy"
    METADATA_FILE = "metadata.parquet"
    MANIFEST_FILE = "manifest.json"

    ANALYZER_NAME = "fa_whitespace"
    TEXT_FIELD = "search_text"
    DOC_INDEX_FIELD = "doc_index"
    PRODUCT_ID_FIELD = "product_id"


    def __init__(
        self,
        processor=None,
    ):
        self.processor = processor
        self.index = None
        self.documents = None


    @classmethod
    def _schema(
        cls,
    ):
        builder = (
            tantivy.SchemaBuilder()
        )

        builder.add_integer_field(
            cls.DOC_INDEX_FIELD,
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
            tokenizer_name=(
                cls.ANALYZER_NAME
            ),
        )

        return builder.build()


    @classmethod
    def _analyzer(
        cls,
    ):
        return (
            tantivy
            .TextAnalyzerBuilder(
                tantivy
                .Tokenizer
                .whitespace()
            )
            .build()
        )


    @classmethod
    def _register(
        cls,
        index,
    ):
        index.register_tokenizer(
            cls.ANALYZER_NAME,
            cls._analyzer(),
        )


    @classmethod
    def build_from_parquet(
        cls,
        input_path,
        output_path,
        processor=None,
        batch_size=50_000,
        writer_heap_size=128_000_000,
        num_threads=1,
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

        parquet_file = (
            pq.ParquetFile(
                input_path
            )
        )

        schema = cls._schema()

        index = tantivy.Index(
            schema,
            path=str(
                tantivy_path
            ),
        )

        cls._register(
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

        total = 0

        for batch in (
            parquet_file
            .iter_batches(
                batch_size=int(
                    batch_size
                ),
                columns=[
                    "id",
                    "search_text",
                ],
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
                texts = texts.map(
                    processor.process
                )

            for local_index, (
                product_id,
                text,
            ) in enumerate(
                zip(
                    frame["id"],
                    texts,
                )
            ):
                writer.add_document(
                    tantivy.Document(
                        doc_index=int(
                            total
                            + local_index
                        ),
                        product_id=int(
                            product_id
                        ),
                        search_text=str(
                            text
                        ),
                    )
                )

            total += len(
                frame
            )

            print(
                "Indexed "
                f"{total:,} products"
            )

        writer.commit()
        writer.wait_merging_threads()
        index.reload()

        # Reuse the canonical product file itself as metadata.
        import shutil as _shutil
        _shutil.copy2(
            input_path,
            output_path
            / cls.METADATA_FILE,
        )

        manifest = {
            "backend": "tantivy",
            "num_documents": int(
                total
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

        self.index = (
            tantivy.Index.open(
                str(
                    path
                    / self.INDEX_SUBDIR
                )
            )
        )

        self._register(
            self.index
        )

        self.documents = (
            pd.read_parquet(
                path
                / self.METADATA_FILE
            )
            .reset_index(drop=True)
        )

        return self


    def _query(
        self,
        query,
    ):
        if self.processor is not None:
            query = (
                self.processor
                .process(
                    query
                )
            )

        terms = [
            token
            for token in str(
                query
            ).split()
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

        return (
            tantivy.Query
            .boolean_query(
                [
                    (
                        tantivy.Occur.Should,
                        term_query,
                    )
                    for term_query
                    in term_queries
                ],
                minimum_number_should_match=1,
            )
        )


    def retrieve(
        self,
        query,
        top_k=10,
    ):
        if self.index is None:
            raise RuntimeError(
                "Product BM25 index "
                "is not loaded"
            )

        parsed = self._query(
            query
        )

        if parsed is None:
            return (
                self.documents
                .iloc[0:0]
                .copy()
            )

        searcher = (
            self.index
            .searcher()
        )

        result = searcher.search(
            parsed,
            limit=int(
                top_k
            ),
            count=False,
        )

        row_ids = []
        scores = []

        for score, address in (
            result.hits
        ):
            document = (
                searcher.doc(
                    address
                )
            )

            value = document[
                self.DOC_INDEX_FIELD
            ]

            if isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            ):
                value = value[0]

            row_ids.append(
                int(value)
            )

            scores.append(
                float(score)
            )

        output = (
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

        output.insert(
            0,
            "doc_index",
            np.asarray(
                row_ids,
                dtype=np.int64,
            ),
        )

        output.insert(
            1,
            "score",
            np.asarray(
                scores,
                dtype=float,
            ),
        )

        return output
