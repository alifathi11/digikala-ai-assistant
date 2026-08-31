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
            <div class="hero-kicker">Manager analytics · Notebooks 11–12</div>
            <h1>Deterministic Catalog Analytics with Grounded LLM Explanations</h1>
            <p>
                Product, price, rating, category, and review KPIs are calculated
                in Python. The LLM does not calculate business numbers — it only
                explains a verified fact map produced by the analytics backend.
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
        "Canonical products",
        "948K",
    )

    columns[1].metric(
        "Comment → product join",
        "100%",
    )

    columns[2].metric(
        "TEST quality",
        "4.86 / 5",
    )

    columns[3].metric(
        "Numeric faithfulness",
        "100%",
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
            "Catalog overview",
            (
                "Managers can inspect catalog size, current price, rating, "
                "review coverage, and engagement signals."
            ),
        )

    with columns[1]:
        _capability_card(
            "02",
            "Category drill-down",
            (
                "Category1 and Category2 filters provide deterministic KPIs "
                "for a selected business scope."
            ),
        )

    with columns[2]:
        _capability_card(
            "03",
            "Category comparison",
            (
                "Two or three categories can be compared using the same "
                "verified metrics and consistent definitions."
            ),
        )

    with columns[3]:
        _capability_card(
            "04",
            "Grounded manager Q&A",
            (
                "The LLM turns verified metrics into concise managerial "
                "explanations without being allowed to invent numbers."
            ),
        )


def _data_readiness():
    st.markdown(
        "## Data readiness"
    )

    readiness = pd.DataFrame(
        {
            "Metric": [
                "Current price",
                "Category2",
                "Review rating",
                "Product rating",
                "Brand",
                "Historical price",
            ],
            "Usable coverage": [
                99.98,
                80.88,
                91.32,
                38.06,
                44.11,
                5.86,
            ],
        }
    ).set_index(
        "Metric"
    )

    left, right = st.columns(
        [1.15, 0.85]
    )

    with left:
        st.bar_chart(
            readiness,
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Ready</strong>
                <p>
                    Product count, current price, category analysis,
                    review coverage, review-volume ranking, and review-rating
                    statistics are production-ready.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <strong>Limited / unavailable</strong>
                <p>
                    Product ratings and brands require coverage caveats.
                    Historical-price analysis is disabled because only
                    <strong>5.86%</strong> of products have usable history.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _backend():
    st.markdown(
        "## Backend"
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">Canonical Products</span>
            <span class="pipeline-chip">Review Aggregates</span>
            <span class="pipeline-chip">Deterministic Python KPIs</span>
            <span class="pipeline-chip">Verified Fact Map</span>
            <span class="pipeline-chip">Grounded LLM</span>
            <span class="pipeline-chip">Numeric Validation</span>
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
                <strong>Python owns the numbers</strong>
                <p>
                    Counts, medians, weighted ratings, coverage, rankings,
                    and category comparisons are computed deterministically
                    before any LLM call.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.markdown(
            """
            <div class="section-card">
                <strong>Verified fact map</strong>
                <p>
                    Only approved facts are exposed to the manager-answer
                    prompt, together with explicit caveats about incomplete
                    data and unsupported business claims.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Numeric hard constraint</strong>
                <p>
                    The LLM references numeric placeholders instead of writing
                    digits directly. The backend validates and renders the final
                    numbers from deterministic facts.
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

    columns[0].metric(
        "Benchmark cases",
        "15",
        "5 DEV · 10 TEST",
        delta_color="off",
    )

    columns[1].metric(
        "Successful runs",
        "15 / 15",
        "0 execution errors",
        delta_color="off",
    )

    columns[2].metric(
        "Evaluation layers",
        "2",
        "Deterministic + LLM judge",
        delta_color="off",
    )

    columns[3].metric(
        "Case coverage",
        "12 types",
        "Overview, policy, comparison, etc.",
        delta_color="off",
    )

    st.markdown(
        """
        <div class="note-box">
            <strong>Evaluation strategy</strong><br><br>
            Deterministic checks verify numeric faithfulness, exact fact values,
            scope counts, category-comparison facts, rendered metrics, and policy
            guards. A separate LLM judge scores the clarity and usefulness of the
            final managerial explanation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _results():
    st.markdown(
        "## Evaluation results"
    )

    quality = pd.DataFrame(
        {
            "Metric": [
                "Numeric faithfulness",
                "Fact accuracy",
                "Scope accuracy",
                "Comparison accuracy",
                "Rendered metrics",
                "Policy guards",
            ],
            "Score (%)": [
                100,
                100,
                100,
                100,
                100,
                100,
            ],
        }
    ).set_index(
        "Metric"
    )

    left, right = st.columns(
        [1.15, 0.85]
    )

    with left:
        st.bar_chart(
            quality,
            use_container_width=True,
        )

    with right:
        st.metric(
            "TEST overall judge score",
            "4.86 / 5",
        )

        st.metric(
            "Managerial usefulness",
            "4.90 / 5",
        )

        st.metric(
            "Relevance",
            "5.00 / 5",
        )

    st.markdown(
        """
        <div class="section-card">
            <strong>Residual errors were qualitative, not numerical</strong>
            <p>
                The main remaining issues were missed requested metrics and a
                small number of unsupported wording choices. All deterministic
                numeric checks remained at <strong>100%</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "TEST mean answer latency: 15.77 s · P95: 23.98 s. "
        "Total answer + judge evaluation cost: approximately $0.207."
    )


def render():
    _hero()
    _headline_metrics()

    st.divider()

    _what_we_built()

    st.divider()

    _data_readiness()

    st.divider()

    _backend()

    st.divider()

    _evaluation_design()

    st.divider()

    _results()
