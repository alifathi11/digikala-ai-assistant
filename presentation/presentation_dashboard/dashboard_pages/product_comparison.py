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
            <div class="hero-kicker">Evidence-based comparison · Notebook 10</div>
            <h1>Compare Products with Review-Grounded Reasoning</h1>
            <p>
                Users select two or three products and specify comparison criteria.
                The system writes a grounded comparison, highlights trade-offs,
                and cites the evidence it used.
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
        "Overall quality",
        "4.94 / 5",
    )

    columns[1].metric(
        "Citation validity",
        "100%",
    )

    columns[2].metric(
        "Deterministic winner accuracy",
        "100%",
    )

    columns[3].metric(
        "No-winner accuracy",
        "100%",
    )


def _card(
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
        _card(
            "01",
            "Two- and three-way comparison",
            (
                "The interface supports side-by-side reasoning for two or "
                "three shortlisted products."
            ),
        )

    with columns[1]:
        _card(
            "02",
            "Criterion-aware analysis",
            (
                "Users can specify the comparison criteria explicitly, such as "
                "oil control, hydration, price, or overall suitability."
            ),
        )

    with columns[2]:
        _card(
            "03",
            "Grounded evidence use",
            (
                "The response is based on product metadata and retrieved "
                "customer-review evidence rather than free-form opinion."
            ),
        )

    with columns[3]:
        _card(
            "04",
            "Calibrated recommendation",
            (
                "When evidence is insufficient or conflicting, the system can "
                "decline to declare a winner instead of forcing one."
            ),
        )


def _backend():
    st.markdown(
        "## Backend"
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">2–3 Selected Products</span>
            <span class="pipeline-chip">Metadata Context</span>
            <span class="pipeline-chip">Review Evidence</span>
            <span class="pipeline-chip">Comparison Pipeline</span>
            <span class="pipeline-chip">Structured Comparison</span>
            <span class="pipeline-chip">LLM Judge</span>
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
                <strong>Context assembly</strong>
                <p>
                    The comparison pipeline gathers the chosen product metadata
                    together with supporting review evidence before prompting the model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.markdown(
            """
            <div class="section-card">
                <strong>Structured reasoning</strong>
                <p>
                    The model compares the products on the requested criteria,
                    produces grounded assessments, and can output an overall winner
                    only when the evidence supports it.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Quality control</strong>
                <p>
                    A separate LLM judge evaluates correctness, groundedness,
                    criterion coverage, calibration, and citation behavior.
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
        "DEV + TEST",
        delta_color="off",
    )

    columns[1].metric(
        "Successful runs",
        "15 / 15",
        "0 execution errors",
        delta_color="off",
    )

    columns[2].metric(
        "Case families",
        "6",
        "Experiential, deterministic, negative, etc.",
        delta_color="off",
    )

    columns[3].metric(
        "Evaluation mode",
        "LLM judge",
        "with deterministic checks",
        delta_color="off",
    )

    st.markdown(
        """
        <div class="note-box">
            <strong>Evaluation strategy</strong><br><br>
            The benchmark mixes experiential comparisons, deterministic metadata
            cases, negative comparisons, three-way comparisons, conflict cases,
            and an insufficient-evidence stress test. The final report emphasizes
            TEST performance but also checks global robustness across all 15 cases.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _results():
    st.markdown(
        "## Evaluation results"
    )

    mode = st.radio(
        "Result view",
        [
            "Quality",
            "Efficiency",
        ],
        horizontal=True,
        key="comparison_result_view",
        label_visibility="collapsed",
    )

    if mode == "Quality":
        data = pd.DataFrame(
            {
                "Dimension": [
                    "Overall",
                    "Correctness",
                    "Groundedness",
                    "Criterion coverage",
                    "Conflict handling",
                    "Calibration",
                ],
                "Score": [
                    4.985,
                    5.000,
                    5.000,
                    5.000,
                    5.000,
                    4.900,
                ],
            }
        ).set_index(
            "Dimension"
        )

        left, right = st.columns(
            [1.25, 0.75]
        )

        with left:
            st.bar_chart(
                data,
                use_container_width=True,
            )

        with right:
            st.metric(
                "TEST overall",
                "4.985 / 5",
            )
            st.metric(
                "Assessment coverage",
                "100%",
            )
            st.metric(
                "Citation ownership",
                "100%",
            )

        st.caption(
            "Across all 15 benchmark cases, the global overall score was 4.943/5."
        )

    else:
        telemetry = pd.DataFrame(
            {
                "Metric": [
                    "Mean comparison latency",
                    "Mean judge latency",
                    "Mean end-to-end latency",
                    "Mean end-to-end tokens",
                    "Estimated total eval cost",
                ],
                "Value": [
                    "11.08 s",
                    "10.01 s",
                    "21.10 s",
                    "4,154",
                    "$0.147",
                ],
            }
        )

        left, right = st.columns(
            [1.0, 1.0]
        )

        with left:
            st.dataframe(
                telemetry,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown(
                """
                <div class="section-card">
                    <strong>Operational takeaway</strong>
                    <p>
                        The comparison feature is high quality and robust, but
                        also the most generation-heavy experience in the project.
                        It trades latency and token cost for stronger reasoning.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "P95 end-to-end latency",
                "30.66 s",
            )

            st.metric(
                "Mean comparison tokens",
                "1,636",
            )


def _notable_takeaway():
    st.markdown(
        "## Key takeaway"
    )

    st.markdown(
        """
        <div class="section-card">
            <strong>Only one notable failure pattern appeared</strong>
            <p>
                The entire benchmark produced just one failure tag:
                <code>insufficient_evidence_mishandled</code>.
                This means the core comparison logic was consistently strong,
                while the main residual risk was over-confident reasoning when
                evidence should have been treated as inconclusive.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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

    st.divider()

    _notable_takeaway()
