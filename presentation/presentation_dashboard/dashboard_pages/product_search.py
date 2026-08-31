from pathlib import Path

import pandas as pd
import streamlit as st


ASSET_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "assets"
)


def _hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Product discovery · Notebooks 08–09</div>
            <h1>Natural-Language Product Search with Grounded Reranking</h1>
            <p>
                Users describe what they need in Persian.
                The system retrieves candidate products from catalog metadata,
                adds review evidence, and uses an LLM reranker to push the most
                relevant products to the top.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _headline_metrics():
    columns = st.columns(
        4
    )

    columns[0].metric(
        "Evaluation queries",
        "30",
        "5 query types",
        delta_color="off",
    )

    columns[1].metric(
        "Judged relevance rows",
        "587",
        "LLM-assisted qrels",
        delta_color="off",
    )

    columns[2].metric(
        "HitRate@1",
        "0.70",
        "+0.30 vs metadata-only",
    )

    columns[3].metric(
        "nDCG@10",
        "0.635",
        "+32% vs metadata-only",
    )


def _capability_card(
    number,
    title,
    text,
):
    st.markdown(
        f"""
        <div class="trace-card">
            <div class="trace-id">Capability {number}</div>
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _what_we_built():
    st.markdown(
        "## What we built"
    )

    columns = st.columns(
        4
    )

    with columns[0]:
        _capability_card(
            "01",
            "Natural-language discovery",
            (
                "Search supports brand, attribute, experiential, negative, "
                "and multi-constraint Persian queries."
            ),
        )

    with columns[1]:
        _capability_card(
            "02",
            "Hybrid metadata retrieval",
            (
                "Dense semantic search and sparse lexical search generate the "
                "initial product candidate pool."
            ),
        )

    with columns[2]:
        _capability_card(
            "03",
            "Review evidence",
            (
                "Candidate products are enriched with product-scoped customer "
                "review evidence before final ranking."
            ),
        )

    with columns[3]:
        _capability_card(
            "04",
            "LLM reranking",
            (
                "A grounded reranker evaluates the strongest candidates and "
                "combines semantic fit with deterministic metadata signals."
            ),
        )


def _backend():
    st.markdown(
        "## Backend"
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">Persian Query</span>
            <span class="pipeline-chip">FAISS + Tantivy</span>
            <span class="pipeline-chip">Metadata Candidates</span>
            <span class="pipeline-chip">Review Evidence</span>
            <span class="pipeline-chip">LLM Reranker</span>
            <span class="pipeline-chip">Final Ranking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns(
        3
    )

    with left:
        st.markdown(
            """
            <div class="section-card">
                <strong>Candidate retrieval</strong>
                <p>
                    Canonical product metadata is searched with dense FAISS and
                    sparse Tantivy retrieval. The production setup starts from a
                    broad metadata candidate pool before reranking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.markdown(
            """
            <div class="section-card">
                <strong>Evidence enrichment</strong>
                <p>
                    Customer-review evidence is retrieved only for shortlisted
                    products, keeping the expensive evidence stage candidate-scoped.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Frozen ranking policy</strong>
                <p>
                    The final production policy is <code>tiered_30_70</code>:
                    deterministic metadata signals are retained while the grounded
                    LLM score drives the final ordering.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Production setup: 50 metadata candidates → 12 reranker candidates → up to 2 review comments per product."
    )


def _evaluation_design():
    st.markdown(
        "## How we evaluated it"
    )

    columns = st.columns(
        3
    )

    columns[0].metric(
        "Query set",
        "30",
        "Balanced across 5 intents",
        delta_color="off",
    )

    columns[1].metric(
        "Relevance pool",
        "587 qrels",
        "Graded 0–3",
        delta_color="off",
    )

    columns[2].metric(
        "Held-out TEST",
        "10 queries",
        "Production policy reported here",
        delta_color="off",
    )

    st.markdown(
        """
        <div class="note-box">
            <strong>Evaluation strategy</strong><br><br>
            Notebook 08 builds a graded relevance pool using an LLM teacher
            and an evaluation-only lexical rescue step. Notebook 09 runs the
            full search pipeline once per query, compares ranking policies,
            freezes <code>tiered_30_70</code>, and reports held-out TEST metrics.
            The qrels are an LLM-assisted proxy, not independent human gold labels.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _results():
    st.markdown(
        "## Evaluation results"
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "HitRate@1",
                "MRR@10",
                "nDCG@10",
            ],
            "Metadata only": [
                0.40,
                0.484286,
                0.480055,
            ],
            "LLM reranked": [
                0.70,
                0.70,
                0.635153,
            ],
        }
    ).set_index(
        "Metric"
    )

    left, right = st.columns(
        [1.2, 0.8]
    )

    with left:
        st.bar_chart(
            comparison,
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Held-out TEST</strong>
                <p>
                    <strong>HitRate@1:</strong> 0.40 → 0.70<br>
                    <strong>MRR@10:</strong> 0.484 → 0.700<br>
                    <strong>nDCG@10:</strong> 0.480 → 0.635
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Candidate Recall@12",
            "62.8%",
        )

        st.metric(
            "Mean search latency",
            "14.71 s",
        )

    st.caption(
        "Across all 30 queries: 19 were classified as OK, 8 had no relevant item in the judged pool, "
        "2 were candidate-retrieval misses, and 1 was a top-rank error."
    )


def render():
    _hero()
    _headline_metrics()

    st.divider()

    _what_we_built()

    st.divider()

    _backend()

    st.divider()

    _evaluation_design()

    st.divider()

    _results()
