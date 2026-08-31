from pathlib import Path

import streamlit as st


ASSET_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "assets"
)

RAW_PRODUCTS = 1_283_496
RAW_COMMENTS = 6_156_289
CLEAN_PRODUCTS = 960_367
CLEAN_COMMENTS = 6_153_060

PRODUCT_RETENTION = (
    CLEAN_PRODUCTS
    / RAW_PRODUCTS
)

COMMENT_RETENTION = (
    CLEAN_COMMENTS
    / RAW_COMMENTS
)


def _hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Data foundation · Notebooks 01–02</div>
            <h1>From Raw Marketplace Data to a Reliable AI Foundation</h1>
            <p>
                Before retrieval or LLM reasoning, the project audits the raw
                Digikala catalog and review corpus, removes structural noise,
                and validates the clean datasets used by every downstream stage.
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
        "Raw products",
        "1.28M",
    )

    columns[1].metric(
        "Raw comments",
        "6.16M",
    )

    columns[2].metric(
        "Clean products",
        "960K",
    )

    columns[3].metric(
        "Clean comments",
        "6.15M",
    )


def _raw_to_clean():
    st.markdown(
        "## Raw → Clean"
    )

    left, right = st.columns(
        2
    )

    with left:
        st.markdown(
            """
            <div class="section-card">
                <strong>Products</strong>
                <p>
                    1,283,496 raw rows → 960,367 clean rows
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            PRODUCT_RETENTION,
            text=(
                f"{PRODUCT_RETENTION * 100:.1f}% retained"
            ),
        )

        st.caption(
            "323,129 duplicate product rows removed."
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <strong>Comments</strong>
                <p>
                    6,156,289 raw rows → 6,153,060 clean rows
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            COMMENT_RETENTION,
            text=(
                f"{COMMENT_RETENTION * 100:.2f}% retained"
            ),
        )

        st.caption(
            "3,229 duplicate comment IDs removed."
        )


def _finding_card(
    *,
    number,
    title,
    text,
):
    st.markdown(
        f"""
        <div class="trace-card">
            <div class="trace-id">Key finding</div>
            <h4>{number}</h4>
            <p>
                <strong>{title}</strong><br>
                {text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _key_findings():
    st.markdown(
        "## What the audit found"
    )

    columns = st.columns(
        4
    )

    with columns[0]:
        _finding_card(
            number="323K",
            title="Duplicate product rows",
            text=(
                "The largest structural cleanup in the raw catalog."
            ),
        )

    with columns[1]:
        _finding_card(
            number="3,229",
            title="Duplicate comment IDs",
            text=(
                "Removed before building retrieval indexes."
            ),
        )

    with columns[2]:
        _finding_card(
            number="16.37%",
            title="Category2 missingness",
            text=(
                "The main product-metadata coverage limitation."
            ),
        )

    with columns[3]:
        _finding_card(
            number="2500",
            title="Raw rating anomaly",
            text=(
                "Notebook 01 exposed out-of-range comment ratings before cleaning."
            ),
        )


def _pipeline_step(
    *,
    index,
    title,
    text,
):
    st.markdown(
        f"""
        <div class="trace-card">
            <div class="trace-id">Step {index}</div>
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preprocessing_story():
    st.markdown(
        "## Preprocessing pipeline"
    )

    st.markdown(
        """
        <div class="pipeline-strip">
            <span class="pipeline-chip">Raw CSV</span>
            <span class="pipeline-chip">Deduplicate</span>
            <span class="pipeline-chip">Clean & normalize</span>
            <span class="pipeline-chip">Validate links & ratings</span>
            <span class="pipeline-chip">Clean Parquet</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(
        4
    )

    with columns[0]:
        _pipeline_step(
            index="01",
            title="Remove duplicates",
            text=(
                "Eliminate repeated product rows and duplicate comment IDs."
            ),
        )

    with columns[1]:
        _pipeline_step(
            index="02",
            title="Normalize fields",
            text=(
                "Prepare consistent values and a Gregorian comment-date field."
            ),
        )

    with columns[2]:
        _pipeline_step(
            index="03",
            title="Enforce integrity",
            text=(
                "Keep valid ratings and ensure each comment maps to a product."
            ),
        )

    with columns[3]:
        _pipeline_step(
            index="04",
            title="Persist clean data",
            text=(
                "Write products_clean.parquet and comments_clean.parquet."
            ),
        )


def _validation_summary():
    st.markdown(
        "## Final validation"
    )

    left, right = st.columns(
        [0.78, 1.22]
    )

    with left:
        st.metric(
            "Sanity checks passed",
            "5 / 5",
            "Ready for indexing",
            delta_color="off",
        )

        st.markdown(
            """
            <div class="note-box">
                <strong>Outcome</strong><br><br>
                The processed datasets form the trusted data boundary for
                embeddings, BM25 retrieval, grounded QA, product search,
                comparison, and analytics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        checks = [
            "No exact duplicate product rows",
            "No exact duplicate comment rows",
            "No duplicate comment IDs",
            "No processed comment rating above 5",
            "No unmatched comment → product links",
        ]

        rows = "".join(
            (
                '<div class="pass-row">'
                f"<span>{check}</span>"
                '<span class="pass-badge">PASS</span>'
                "</div>"
            )
            for check in checks
        )

        st.markdown(
            (
                '<div class="section-card">'
                f"{rows}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _notebook_visual():
    st.markdown(
        "## One visual from the raw-data audit"
    )

    choice = st.selectbox(
        "Choose the notebook view",
        [
            "Product rating distribution",
            "Comment rating distribution",
            "Review text length",
        ],
        label_visibility="collapsed",
    )

    mapping = {
        "Product rating distribution": (
            "product_rating_distribution.png",
            "Raw product-rating distribution inspected before preprocessing.",
        ),
        "Comment rating distribution": (
            "comment_rating_distribution.png",
            "Raw comment ratings reveal the need for rating-range validation.",
        ),
        "Review text length": (
            "review_body_length_distribution.png",
            "Review text is typically short, with a long tail of larger comments.",
        ),
    }

    filename, caption = mapping[
        choice
    ]

    st.image(
        str(
            ASSET_DIR
            / filename
        ),
        caption=caption,
        use_container_width=True,
    )


def render():
    _hero()
    _headline_metrics()

    st.divider()

    _raw_to_clean()

    st.divider()

    _key_findings()

    st.divider()

    _preprocessing_story()

    st.divider()

    _validation_summary()

    st.divider()

    _notebook_visual()
