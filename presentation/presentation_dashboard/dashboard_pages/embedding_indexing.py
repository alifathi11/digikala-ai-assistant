import html

import pandas as pd
import streamlit as st


COMMENT_DOCS = 6_153_060
PRODUCT_DOCS = 948_352
VECTOR_DIMENSION = 384

COMMENT_DENSE_MINUTES = 16.81
COMMENT_SPARSE_MINUTES = 1.54
PRODUCT_DENSE_MINUTES = 3.77
PRODUCT_SPARSE_MINUTES = 0.14

COMMENT_DENSE_THROUGHPUT = (
    COMMENT_DOCS
    / (
        COMMENT_DENSE_MINUTES
        * 60
    )
)

COMMENT_SPARSE_THROUGHPUT = (
    COMMENT_DOCS
    / (
        COMMENT_SPARSE_MINUTES
        * 60
    )
)

PRODUCT_DENSE_THROUGHPUT = (
    PRODUCT_DOCS
    / (
        PRODUCT_DENSE_MINUTES
        * 60
    )
)

PRODUCT_SPARSE_THROUGHPUT = (
    PRODUCT_DOCS
    / (
        PRODUCT_SPARSE_MINUTES
        * 60
    )
)


def _safe(
    value,
):
    return html.escape(
        str(value),
        quote=True,
    )


def _hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Retrieval foundation · Notebooks 03, 04 & 07</div>
            <h1>Embedding & Search Index Construction</h1>
            <p>
                This page summarizes how the project converts processed comments
                and canonical product metadata into persistent dense and sparse
                search indexes. Dense retrieval is backed by FAISS embeddings;
                sparse lexical retrieval is backed by Tantivy BM25.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">Processed Parquet</span>
            <span class="pipeline-chip">TextProcessor</span>
            <span class="pipeline-chip">384-D embeddings</span>
            <span class="pipeline-chip">FAISS</span>
            <span class="pipeline-chip">Tantivy BM25</span>
            <span class="pipeline-chip">Persistent metadata</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _top_metrics():
    columns = st.columns(
        4
    )

    columns[0].metric(
        "Comment documents indexed",
        f"{COMMENT_DOCS:,}",
    )

    columns[1].metric(
        "Canonical products indexed",
        f"{PRODUCT_DOCS:,}",
    )

    columns[2].metric(
        "Embedding dimension",
        f"{VECTOR_DIMENSION}",
    )

    columns[3].metric(
        "Index families",
        "2",
        "Dense + Sparse",
        delta_color="off",
    )


def _architecture():
    st.subheader(
        "Indexing architecture"
    )

    entity = st.radio(
        "Inspect indexing path",
        [
            "Comments",
            "Products",
        ],
        horizontal=True,
        key="index_architecture_entity",
    )

    if entity == "Comments":
        st.markdown(
            """
            <div class="section-card">
                <strong>Comment retrieval path</strong>
                <p>
                    <code>comments_clean.parquet</code> is normalized by the shared
                    <code>TextProcessor</code>. The same processed corpus is then
                    indexed twice: once as dense vectors in FAISS and once as lexical
                    text in Tantivy. Both indexes persist metadata needed to map
                    retrieval results back to comment records.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        flow = pd.DataFrame(
            [
                (
                    "1",
                    "Input",
                    "comments_clean.parquet",
                    f"{COMMENT_DOCS:,} processed comments",
                ),
                (
                    "2",
                    "Text processing",
                    "TextProcessor",
                    "Shared text normalization before indexing",
                ),
                (
                    "3A",
                    "Dense branch",
                    "EmbeddingFactory → FAISS",
                    "384-D vector representation",
                ),
                (
                    "3B",
                    "Sparse branch",
                    "Tantivy BM25",
                    "Lexical search_text representation",
                ),
                (
                    "4",
                    "Persistence",
                    "Index + metadata.parquet",
                    "Disk-backed retrieval artifacts",
                ),
            ],
            columns=[
                "Step",
                "Layer",
                "Implementation",
                "Purpose",
            ],
        )

    else:
        st.markdown(
            """
            <div class="section-card">
                <strong>Product retrieval path</strong>
                <p>
                    Product indexing starts with an explicit canonicalization step:
                    multiple source rows are reduced to one row per product ID and
                    written to <code>products_search.parquet</code>. The canonical
                    search table is then indexed independently by FAISS and Tantivy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        flow = pd.DataFrame(
            [
                (
                    "1",
                    "Input",
                    "products_clean.parquet",
                    "Processed product rows",
                ),
                (
                    "2",
                    "Canonicalization",
                    "build_canonical_products_file",
                    "One searchable row per product ID",
                ),
                (
                    "3",
                    "Search table",
                    "products_search.parquet",
                    f"{PRODUCT_DOCS:,} unique product IDs",
                ),
                (
                    "4A",
                    "Dense branch",
                    "EmbeddingFactory → ProductFAISSIndex",
                    "384-D product metadata vectors",
                ),
                (
                    "4B",
                    "Sparse branch",
                    "ProductBM25Index / Tantivy",
                    "Lexical product metadata index",
                ),
            ],
            columns=[
                "Step",
                "Layer",
                "Implementation",
                "Purpose",
            ],
        )

    st.dataframe(
        flow,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="note-box">
            <strong>Design choice</strong><br><br>
            Dense and sparse indexes are built separately instead of forcing one
            retrieval representation to handle every query type. The notebooks here
            only build the persistent retrieval artifacts; ranking and evaluation
            happen in later project stages.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _index_card(
    *,
    label,
    title,
    body,
):
    st.markdown(
        f"""
        <div class="trace-card">
            <div class="trace-id">{_safe(label)}</div>
            <h4>{_safe(title)}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _comment_index():
    st.subheader(
        "Comment indexes"
    )

    dense_tab, sparse_tab = st.tabs(
        [
            "Dense · FAISS",
            "Sparse · Tantivy BM25",
        ]
    )

    with dense_tab:
        columns = st.columns(
            4
        )

        columns[0].metric(
            "Documents",
            f"{COMMENT_DOCS:,}",
        )

        columns[1].metric(
            "Vector dimension",
            "384",
        )

        columns[2].metric(
            "Build time",
            "16.81 min",
        )

        columns[3].metric(
            "Observed throughput",
            f"{COMMENT_DENSE_THROUGHPUT:,.0f} docs/s",
        )

        left, right = st.columns(
            2
        )

        with left:
            _index_card(
                label="Backend",
                title="FAISS IndexFlatIP",
                body=(
                    "The executed manifest reports <code>IndexFlatIP</code>. "
                    "Embeddings are normalized, so inner-product search operates "
                    "on normalized vector representations."
                ),
            )

        with right:
            _index_card(
                label="Embedding pipeline",
                title="Configured multilingual encoder",
                body=(
                    "The notebook constructs the encoder through "
                    "<code>EmbeddingFactory.create()</code> using the central "
                    "embedding provider/model configuration. The produced vectors "
                    "are <strong>384-dimensional</strong>."
                ),
            )

        st.markdown(
            "### Executed build configuration"
        )

        config = pd.DataFrame(
            [
                (
                    "Input",
                    "data/processed/comments_clean.parquet",
                ),
                (
                    "Output",
                    "data/indexes/product_comments_embedding",
                ),
                (
                    "Index type",
                    "IndexFlatIP",
                ),
                (
                    "Chunk size",
                    "5,000 documents",
                ),
                (
                    "Encode batch size",
                    "64",
                ),
                (
                    "Normalized embeddings",
                    "True",
                ),
                (
                    "Metadata artifact",
                    "metadata.parquet",
                ),
                (
                    "Overwrite",
                    "True",
                ),
            ],
            columns=[
                "Parameter",
                "Executed value",
            ],
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The build processed all 6,153,060 comments and persisted the dense index plus metadata."
        )

    with sparse_tab:
        columns = st.columns(
            4
        )

        columns[0].metric(
            "Documents",
            f"{COMMENT_DOCS:,}",
        )

        columns[1].metric(
            "Backend",
            "Tantivy",
        )

        columns[2].metric(
            "Build time",
            "1.54 min",
        )

        columns[3].metric(
            "Observed throughput",
            f"{COMMENT_SPARSE_THROUGHPUT:,.0f} docs/s",
        )

        left, right = st.columns(
            2
        )

        with left:
            _index_card(
                label="Sparse retrieval",
                title="Tantivy BM25",
                body=(
                    "The global lexical index stores the processed "
                    "<code>search_text</code> field and uses a whitespace tokenizer. "
                    "It is persisted to disk for sparse retrieval."
                ),
            )

        with right:
            _index_card(
                label="Memory control",
                title="Bounded indexing settings",
                body=(
                    "Notebook 04 indexes in 50,000-document batches with a "
                    "128,000,000-byte writer heap, one indexing thread, and a "
                    "commit every 10 batches."
                ),
            )

        config = pd.DataFrame(
            [
                (
                    "Input",
                    "data/processed/comments_clean.parquet",
                ),
                (
                    "Output",
                    "data/indexes/product_comments_bm25_tantivy",
                ),
                (
                    "Tantivy",
                    "v0.26.0 · index format v7",
                ),
                (
                    "Batch size",
                    "50,000 documents",
                ),
                (
                    "Writer heap",
                    "128,000,000 bytes",
                ),
                (
                    "Threads",
                    "1",
                ),
                (
                    "Commit interval",
                    "Every 10 batches",
                ),
                (
                    "Tokenizer",
                    "whitespace",
                ),
                (
                    "Indexed field",
                    "search_text",
                ),
                (
                    "Metadata artifact",
                    "metadata.parquet",
                ),
            ],
            columns=[
                "Parameter",
                "Executed value",
            ],
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )


def _product_index():
    st.subheader(
        "Product indexes"
    )

    st.markdown(
        """
        <div class="section-card">
            <strong>Canonicalization before indexing</strong>
            <p>
                Notebook 07 first creates <code>products_search.parquet</code>.
                The executed output contains <strong>948,352 rows</strong> and
                <strong>948,352 unique product IDs</strong>, establishing a strict
                one-row-per-product search table before either retrieval index is built.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    canonical = pd.DataFrame(
        [
            (
                "Canonical rows",
                948_352,
            ),
            (
                "Unique product IDs",
                948_352,
            ),
        ],
        columns=[
            "Check",
            "Count",
        ],
    ).set_index(
        "Check"
    )

    st.bar_chart(
        canonical,
        use_container_width=True,
    )

    dense_tab, sparse_tab = st.tabs(
        [
            "Dense · FAISS",
            "Sparse · Tantivy BM25",
        ]
    )

    with dense_tab:
        columns = st.columns(
            4
        )

        columns[0].metric(
            "Products",
            f"{PRODUCT_DOCS:,}",
        )

        columns[1].metric(
            "Vector dimension",
            "384",
        )

        columns[2].metric(
            "Build time",
            "3.77 min",
        )

        columns[3].metric(
            "Observed throughput",
            f"{PRODUCT_DENSE_THROUGHPUT:,.0f} products/s",
        )

        config = pd.DataFrame(
            [
                (
                    "Input",
                    "data/processed/products_search.parquet",
                ),
                (
                    "Output",
                    "data/indexes/products_embedding",
                ),
                (
                    "Backend",
                    "FAISS",
                ),
                (
                    "Documents",
                    "948,352",
                ),
                (
                    "Dimension",
                    "384",
                ),
                (
                    "Chunk size",
                    "10,000 products",
                ),
                (
                    "Embedding config",
                    "Central project configuration via EmbeddingFactory",
                ),
            ],
            columns=[
                "Parameter",
                "Executed value",
            ],
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The product dense manifest reports backend, document count, dimension, and chunk size."
        )

    with sparse_tab:
        columns = st.columns(
            4
        )

        columns[0].metric(
            "Products",
            f"{PRODUCT_DOCS:,}",
        )

        columns[1].metric(
            "Backend",
            "Tantivy",
        )

        columns[2].metric(
            "Build time",
            "0.14 min",
        )

        columns[3].metric(
            "Observed throughput",
            f"{PRODUCT_SPARSE_THROUGHPUT:,.0f} products/s",
        )

        config = pd.DataFrame(
            [
                (
                    "Input",
                    "data/processed/products_search.parquet",
                ),
                (
                    "Output",
                    "data/indexes/products_bm25_tantivy",
                ),
                (
                    "Backend",
                    "Tantivy",
                ),
                (
                    "Documents",
                    "948,352",
                ),
                (
                    "Indexing settings",
                    "Loaded from product_search.indexing in central config",
                ),
                (
                    "Overwrite",
                    "True",
                ),
            ],
            columns=[
                "Parameter",
                "Executed value",
            ],
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Notebook 07 reports a completed Tantivy index over all 948,352 canonical products."
        )


def _comparison():
    st.subheader(
        "Build comparison"
    )

    mode = st.segmented_control(
        "Compare by",
        options=[
            "Build time",
            "Throughput",
        ],
        default="Build time",
        key="index_build_comparison",
    )

    data = pd.DataFrame(
        [
            (
                "Comment · FAISS",
                COMMENT_DENSE_MINUTES,
                COMMENT_DENSE_THROUGHPUT,
            ),
            (
                "Comment · Tantivy",
                COMMENT_SPARSE_MINUTES,
                COMMENT_SPARSE_THROUGHPUT,
            ),
            (
                "Product · FAISS",
                PRODUCT_DENSE_MINUTES,
                PRODUCT_DENSE_THROUGHPUT,
            ),
            (
                "Product · Tantivy",
                PRODUCT_SPARSE_MINUTES,
                PRODUCT_SPARSE_THROUGHPUT,
            ),
        ],
        columns=[
            "Index",
            "Build time (min)",
            "Throughput (docs/s)",
        ],
    )

    if mode == "Throughput":
        chart = data[
            [
                "Index",
                "Throughput (docs/s)",
            ]
        ].set_index(
            "Index"
        )
    else:
        chart = data[
            [
                "Index",
                "Build time (min)",
            ]
        ].set_index(
            "Index"
        )

    st.bar_chart(
        chart,
        use_container_width=True,
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Throughput is calculated from the executed document counts and elapsed times printed by the notebooks."
    )

    st.markdown(
        "### Technical traceability"
    )

    trace = pd.DataFrame(
        [
            (
                "03_build_comment_embedding_index.ipynb",
                "Comment dense index",
                "FAISS · 6,153,060 vectors · 384-D",
                "16.81 min",
            ),
            (
                "04_build_comment_bm25_index.ipynb",
                "Comment sparse index",
                "Tantivy BM25 · 6,153,060 docs",
                "1.54 min",
            ),
            (
                "07_build_product_search_indexes.ipynb",
                "Product canonicalization + dense + sparse",
                "948,352 canonical products · FAISS + Tantivy",
                "3.77 + 0.14 min",
            ),
        ],
        columns=[
            "Notebook",
            "Responsibility",
            "Executed artifact",
            "Build time",
        ],
    )

    st.dataframe(
        trace,
        use_container_width=True,
        hide_index=True,
    )


def render():
    _hero()
    _top_metrics()

    tabs = st.tabs(
        [
            "Architecture",
            "Comment Index",
            "Product Index",
            "Build Comparison",
        ]
    )

    with tabs[0]:
        _architecture()

    with tabs[1]:
        _comment_index()

    with tabs[2]:
        _product_index()

    with tabs[3]:
        _comparison()
