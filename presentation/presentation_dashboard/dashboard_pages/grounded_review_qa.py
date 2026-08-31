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
            <div class="hero-kicker">Grounded AI · Notebooks 05–06</div>
            <h1>Ask Questions Directly from Real Customer Reviews</h1>
            <p>
                The user selects a product and asks a question.
                The system retrieves the most relevant reviews, generates a
                grounded answer, and keeps the supporting evidence traceable.
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
        "QA quality",
        "4.94 / 5",
    )

    columns[1].metric(
        "Citation validity",
        "100%",
    )

    columns[2].metric(
        "Hybrid Recall@5",
        "89.0%",
    )

    columns[3].metric(
        "Average QA latency",
        "3.00 s",
    )


def _feature_card(
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
        _feature_card(
            "01",
            "Product-scoped retrieval",
            (
                "Search is restricted to reviews belonging to the selected "
                "product, keeping evidence relevant to the user's context."
            ),
        )

    with columns[1]:
        _feature_card(
            "02",
            "Hybrid search",
            (
                "Lexical BM25 and semantic embeddings are combined to improve "
                "retrieval across both exact wording and meaning."
            ),
        )

    with columns[2]:
        _feature_card(
            "03",
            "Grounded generation",
            (
                "The LLM answers from retrieved review evidence instead of "
                "reasoning over the full corpus."
            ),
        )

    with columns[3]:
        _feature_card(
            "04",
            "Evidence validation",
            (
                "Returned evidence IDs are checked so the UI can expose "
                "traceable support for the generated answer."
            ),
        )


def _backend():
    st.markdown(
        "## Backend"
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">Selected Product</span>
            <span class="pipeline-chip">Tantivy BM25</span>
            <span class="pipeline-chip">FAISS Embeddings</span>
            <span class="pipeline-chip">Hybrid Retriever</span>
            <span class="pipeline-chip">GroundedQAPipeline</span>
            <span class="pipeline-chip">Structured JSON Answer</span>
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
                <strong>Retrieval</strong>
                <p>
                    Tantivy provides sparse BM25 retrieval while FAISS provides
                    dense semantic retrieval. Notebook 05 evaluates both
                    individually and as a hybrid retriever.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.markdown(
            """
            <div class="section-card">
                <strong>Generation</strong>
                <p>
                    <code>GroundedQAPipeline</code> sends the retrieved comment
                    context to an OpenAI-compatible JSON generator through the
                    configured API endpoint.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Grounding</strong>
                <p>
                    The answer is returned together with evidence references,
                    allowing citation validity and evidence quality to be
                    evaluated explicitly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _evaluation_design():
    st.markdown(
        "## How we evaluated it"
    )

    columns = st.columns(
        4
    )

    with columns[0]:
        st.metric(
            "Retrieval benchmark",
            "150 queries",
            "50 products",
            delta_color="off",
        )

    with columns[1]:
        st.metric(
            "Retrievers compared",
            "3",
            "BM25 · Embedding · Hybrid",
            delta_color="off",
        )

    with columns[2]:
        st.metric(
            "QA evaluation",
            "30 samples",
            "Fixed evaluation run",
            delta_color="off",
        )

    with columns[3]:
        st.metric(
            "Evaluation layers",
            "2",
            "Retrieval + Answer quality",
            delta_color="off",
        )

    st.markdown(
        """
        <div class="note-box">
            <strong>Evaluation strategy</strong><br><br>
            Notebook 05 measures ranking quality and latency for BM25,
            embedding, and hybrid retrieval. Notebook 06 then evaluates the
            end-to-end grounded answer using an LLM judge plus deterministic
            citation and evidence metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _retrieval_results():
    st.markdown(
        "### Retrieval result"
    )

    retrieval = pd.DataFrame(
        {
            "Retriever": [
                "BM25",
                "Embedding",
                "Hybrid",
            ],
            "Recall@5": [
                0.8500,
                0.8522,
                0.8900,
            ],
            "nDCG@5": [
                0.7499,
                0.7280,
                0.7883,
            ],
        }
    ).set_index(
        "Retriever"
    )

    left, right = st.columns(
        [1.2, 0.8]
    )

    with left:
        st.bar_chart(
            retrieval,
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Hybrid retrieval performed best</strong>
                <p>
                    Recall@5 reached <strong>89.0%</strong>,
                    nDCG@5 reached <strong>0.788</strong>, and
                    HitRate@5 reached <strong>93.3%</strong>.
                </p>
                <p>
                    Mean hybrid retrieval latency:
                    <strong>264 ms</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _qa_results():
    st.markdown(
        "### End-to-end QA result"
    )

    judge_scores = pd.DataFrame(
        {
            "Dimension": [
                "Correctness",
                "Relevance",
                "Completeness",
                "Groundedness",
                "Instruction following",
                "Safety",
            ],
            "Score": [
                4.93,
                4.97,
                4.83,
                4.97,
                5.00,
                5.00,
            ],
        }
    ).set_index(
        "Dimension"
    )

    left, right = st.columns(
        [1.2, 0.8]
    )

    with left:
        st.bar_chart(
            judge_scores,
            use_container_width=True,
        )

    with right:
        metrics = st.columns(
            2
        )

        metrics[0].metric(
            "Overall",
            "4.94 / 5",
        )

        metrics[1].metric(
            "Groundedness",
            "4.97 / 5",
        )

        metrics[0].metric(
            "Evidence recall",
            "87.8%",
        )

        metrics[1].metric(
            "Retrieval evidence recall",
            "90.0%",
        )

    st.caption(
        "Residual failures were limited: the most common tag was missed key evidence (3 occurrences); "
        "conflict handling appeared in 2 tags and unsupported claim in 1."
    )


def _results():
    st.markdown(
        "## Evaluation results"
    )

    mode = st.radio(
        "Result view",
        [
            "Retrieval",
            "Grounded QA",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="grounded_qa_result_view",
    )

    if mode == "Retrieval":
        _retrieval_results()
    else:
        _qa_results()


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
