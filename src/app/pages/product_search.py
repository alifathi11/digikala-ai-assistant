import html
import time
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.app.safety import render_ui_error


STATUS_LABELS = {
    "support": (
        "شواهد موافق"
    ),
    "mixed": (
        "شواهد ترکیبی"
    ),
    "contradict": (
        "شواهد مخالف"
    ),
    "none": (
        "بدون شاهد مستقیم"
    ),
}


def _escape(
    value,
):
    return html.escape(
        str(value),
        quote=True,
    )


def _safe_value(
    row,
    field,
    default="—",
):
    if isinstance(
        row,
        dict,
    ):
        value = row.get(
            field
        )
    else:
        value = getattr(
            row,
            field,
            None,
        )

    if value is None:
        return default

    try:
        if pd.isna(
            value
        ):
            return default
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def _format_price(
    value,
):
    try:
        if (
            value is None
            or pd.isna(
                value
            )
        ):
            return "—"

        return (
            f"{float(value):,.0f}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def _score_percent(
    value,
):
    try:
        return (
            max(
                0.0,
                min(
                    1.0,
                    float(
                        value
                    ),
                ),
            )
            * 100
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _evidence_documents(
    services,
    evidence_ids,
):
    if not evidence_ids:
        return (
            services
            .retrieval
            .documents
            .iloc[0:0]
            .copy()
        )

    ids = []

    for value in (
        evidence_ids
    ):
        try:
            ids.append(
                int(
                    value
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    if not ids:
        return (
            services
            .retrieval
            .documents
            .iloc[0:0]
            .copy()
        )

    vector_store = (
        services
        .retrieval
        .vector_store
    )

    row_map = getattr(
        vector_store,
        "_comment_id_to_row",
        None,
    )

    if row_map is not None:
        row_ids = [
            row_map[
                comment_id
            ]
            for comment_id
            in ids
            if comment_id
            in row_map
        ]

        if row_ids:
            frame = (
                services
                .retrieval
                .documents
                .iloc[
                    row_ids
                ]
                .copy()
            )

            order = {
                comment_id: rank
                for rank, comment_id
                in enumerate(
                    ids
                )
            }

            frame[
                "_rank"
            ] = (
                frame[
                    "id"
                ]
                .astype(int)
                .map(
                    order
                )
            )

            return (
                frame
                .sort_values(
                    "_rank"
                )
                .drop(
                    columns=[
                        "_rank"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

    return (
        services
        .retrieval
        .documents[
            services
            .retrieval
            .documents[
                "id"
            ]
            .astype(int)
            .isin(
                ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def _render_evidence(
    services,
    evidence_ids,
):
    evidence = (
        _evidence_documents(
            services,
            evidence_ids,
        )
    )

    if len(
        evidence
    ) == 0:
        st.caption(
            "برای این محصول شاهد مستقیمی "
            "از نظرات انتخاب نشده است."
        )
        return

    for row in (
        evidence
        .itertuples(
            index=False
        )
    ):
        comment_id = int(
            getattr(
                row,
                "id"
            )
        )

        rate = getattr(
            row,
            "rate",
            "—",
        )

        body = getattr(
            row,
            "body",
            "",
        )

        if body is None:
            body = ""

        evidence_html = dedent(
            f"""
            <div class="evidence-card">
                <div class="evidence-meta">
                    <span>نظر #{comment_id}</span>
                    <span>·</span>
                    <span>امتیاز {_escape(rate)}</span>
                </div>
                <div class="evidence-body">{_escape(body)}</div>
            </div>
            """
        ).strip()

        st.html(
            evidence_html
        )


def _render_result_card(
    services,
    row,
    rank,
):
    title = _safe_value(
        row,
        "title_fa",
        "بدون عنوان",
    )

    product_id = int(
        _safe_value(
            row,
            "id",
            -1,
        )
    )

    brand = _safe_value(
        row,
        "Brand",
    )

    category = _safe_value(
        row,
        "Category2",
        _safe_value(
            row,
            "Category1",
        ),
    )

    price = _format_price(
        _safe_value(
            row,
            "Price",
            None,
        )
    )

    rate = _safe_value(
        row,
        "Rate",
    )

    reason = _safe_value(
        row,
        "reason",
        "",
    )

    status = str(
        _safe_value(
            row,
            "evidence_status",
            "none",
        )
    ).strip().lower()

    if status not in (
        STATUS_LABELS
    ):
        status = "none"

    status_label = (
        STATUS_LABELS[
            status
        ]
    )

    final_score = (
        _score_percent(
            _safe_value(
                row,
                "score",
                0,
            )
        )
    )

    metadata_score = (
        _score_percent(
            _safe_value(
                row,
                "metadata_score_norm",
                _safe_value(
                    row,
                    "metadata_score",
                    0,
                ),
            )
        )
    )

    llm_score = (
        float(
            _safe_value(
                row,
                "llm_match_score",
                0,
            )
        )
    )

    brand_match = int(
        float(
            _safe_value(
                row,
                "brand_match",
                0,
            )
        )
        > 0
    )

    evidence_ids = (
        _safe_value(
            row,
            "evidence_ids",
            [],
        )
    )

    if not isinstance(
        evidence_ids,
        list,
    ):
        evidence_ids = []

    brand_chip = (
        '<span class="meta-chip brand-match-chip">'
        'تطابق مستقیم برند'
        '</span>'
        if brand_match
        else ""
    )

    card_html = dedent(
        f"""
        <div class="search-result-card">
            <div class="search-result-head">
                <div class="search-rank">#{rank}</div>
                <div class="search-result-title-wrap">
                    <div class="search-result-title">{_escape(title)}</div>
                    <div class="search-result-subtitle">شناسه محصول {product_id}</div>
                </div>
                <div class="search-final-score">
                    <span>امتیاز نهایی</span>
                    <strong>{final_score:.0f}</strong>
                </div>
            </div>
            <div class="product-meta search-result-meta">
                <span class="meta-chip"><strong>برند</strong> {_escape(brand)}</span>
                <span class="meta-chip"><strong>دسته</strong> {_escape(category)}</span>
                <span class="meta-chip"><strong>امتیاز</strong> {_escape(rate)}</span>
                <span class="meta-chip"><strong>قیمت</strong> {_escape(price)}</span>
                {brand_chip}
            </div>
            <div class="search-score-grid">
                <div class="score-stat">
                    <span>تطابق اطلاعات محصول</span>
                    <strong>{metadata_score:.0f}%</strong>
                </div>
                <div class="score-stat">
                    <span>ارزیابی LLM</span>
                    <strong>{llm_score:.0f}/5</strong>
                </div>
                <div class="score-stat status-stat status-{status}">
                    <span>وضعیت شواهد</span>
                    <strong>{_escape(status_label)}</strong>
                </div>
            </div>
            <div class="search-reason">
                <span class="search-reason-label">چرا این محصول؟</span>
                <div>{_escape(reason) if reason else "توضیح مستقیمی ثبت نشده است."}</div>
            </div>
        </div>
        """
    ).strip()

    st.html(
        card_html
    )

    with st.expander(
        (
            "نظرات استنادی"
            f" · {len(evidence_ids)} شاهد"
        )
    ):
        _render_evidence(
            services,
            evidence_ids,
        )


def _render_telemetry(
    telemetry,
):
    col1, col2, col3, col4 = (
        st.columns(4)
    )

    total_latency = float(
        telemetry.get(
            "ui_total_latency_ms",
            0,
        )
        or 0
    )

    llm_latency = float(
        telemetry.get(
            "latency_ms",
            0,
        )
        or 0
    )

    review_latency = float(
        telemetry.get(
            "review_retrieval_latency_ms",
            0,
        )
        or 0
    )

    review_count = int(
        telemetry.get(
            "retrieved_review_count",
            0,
        )
        or 0
    )

    col1.metric(
        "زمان کل",
        (
            f"{total_latency / 1000:.2f} s"
        ),
    )

    col2.metric(
        "LLM",
        (
            f"{llm_latency / 1000:.2f} s"
        ),
    )

    col3.metric(
        "بازیابی نظرات",
        (
            f"{review_latency:.0f} ms"
            if review_latency
            else "—"
        ),
    )

    col4.metric(
        "شواهد بازیابی‌شده",
        str(
            review_count
        )
        if review_count
        else "—",
    )

    tokens = telemetry.get(
        "total_tokens",
        0,
    )

    model = telemetry.get(
        "model",
        "—",
    )

    review_scope = telemetry.get(
        "review_scope"
    )

    details = [
        (
            "توکن‌ها: "
            f"{tokens}"
        ),
        (
            "مدل: "
            f"{model}"
        ),
    ]

    if review_scope:
        details.append(
            (
                "دامنه‌ی بازیابی نظر: "
                "هر محصول به‌صورت مستقل"
                if review_scope
                == "candidate_product"
                else str(
                    review_scope
                )
            )
        )

    st.caption(
        " · ".join(
            details
        )
    )


def render(
    services,
):
    hero_html = dedent(
        """
        <div class="hero search-hero">
            <div class="hero-copy">
                <div class="hero-kicker">● کشف و پیشنهاد محصول</div>
                <h1>محصول مناسب را با زبان خودتان پیدا کنید</h1>
                <p>
                    نیازتان را طبیعی بنویسید؛ جست‌وجوی ترکیبی اطلاعات محصول
                    را پیدا می‌کند و بازرتبه‌بندی هوشمند، نظرات کاربران را
                    برای تشخیص شواهد موافق یا مخالف بررسی می‌کند.
                </p>
            </div>
            <div class="hero-icon">⌕</div>
        </div>
        """
    ).strip()

    st.html(
        hero_html
    )

    with st.form(
        "product_search_form",
        clear_on_submit=False,
    ):
        col_query, col_count = (
            st.columns(
                [
                    5,
                    1,
                ]
            )
        )

        with col_query:
            query = st.text_input(
                "چه محصولی می‌خواهید؟",
                placeholder=(
                    "مثلاً: ضدآفتاب مناسب پوست چرب که جوش نزنه"
                ),
            )

        with col_count:
            top_k = st.selectbox(
                "تعداد نتیجه",
                options=[
                    5,
                    8,
                    10,
                ],
                index=1,
            )

        submitted = (
            st.form_submit_button(
                "جست‌وجوی هوشمند",
                use_container_width=True,
            )
        )

    hint_html = dedent(
        """
        <div class="hint-card search-hint-card">
            <span class="hint-icon">✦</span>
            <span>
                بهتر است نیاز، ویژگی یا تجربه‌ی موردنظرتان را بنویسید؛
                مثال: «شامپو ضد ریزش سریتا» یا «کرم آبرسان سبک و زود جذب».
            </span>
        </div>
        """
    ).strip()

    st.html(
        hint_html
    )

    if submitted:
        normalized_query = str(
            query
        ).strip()

        if not normalized_query:
            st.warning(
                "عبارت جست‌وجو را وارد کنید."
            )
        else:
            try:
                with st.spinner(
                    "در حال جست‌وجو و بررسی نظرات کاربران..."
                ):
                    start = (
                        time.perf_counter()
                    )

                    results = (
                        services
                        .product_search
                        .search(
                            query=(
                                normalized_query
                            ),
                            top_k=int(
                                top_k
                            ),
                        )
                    )

                    elapsed_ms = (
                        time.perf_counter()
                        - start
                    ) * 1000

                if not isinstance(
                    results,
                    pd.DataFrame,
                ):
                    raise ValueError(
                        "خروجی جست‌وجو ساختار جدولی قابل نمایش ندارد."
                    )

                telemetry = dict(
                    results.attrs.get(
                        "telemetry",
                        {},
                    )
                )

                telemetry[
                    "ui_total_latency_ms"
                ] = float(
                    elapsed_ms
                )

                st.session_state[
                    "product_search_result"
                ] = {
                    "query": (
                        normalized_query
                    ),
                    "results": (
                        results
                        .copy()
                    ),
                    "telemetry": (
                        telemetry
                    ),
                }

            except Exception as exc:
                st.session_state.pop(
                    "product_search_result",
                    None,
                )
                render_ui_error(
                    "جست‌وجوی هوشمند نتیجه‌ی قابل نمایش برنگرداند.",
                    exc,
                )

    state = st.session_state.get(
        "product_search_result"
    )

    if state is None:
        st.info(
            "برای شروع، نیازتان را "
            "در کادر بالا جست‌وجو کنید."
        )
        return

    results = state[
        "results"
    ]

    if len(
        results
    ) == 0:
        st.warning(
            "برای این عبارت نتیجه‌ای "
            "پیدا نشد."
        )
        return

    results_header_html = dedent(
        f"""
        <div class="search-results-header">
            <div>
                <span class="search-results-kicker">نتایج برای</span>
                <h2>«{_escape(state["query"])}»</h2>
            </div>
            <div class="search-result-count">{len(results)} محصول</div>
        </div>
        """
    ).strip()

    st.html(
        results_header_html
    )

    rendered_count = 0

    for rank, row in enumerate(
        results.itertuples(
            index=False
        ),
        start=1,
    ):
        try:
            _render_result_card(
                services=services,
                row=row,
                rank=rank,
            )
            rendered_count += 1
        except Exception as exc:
            render_ui_error(
                f"نتیجه‌ی شماره {rank} قابل نمایش نبود و رد شد.",
                exc,
                retry_hint=False,
            )

    if rendered_count == 0:
        st.session_state.pop(
            "product_search_result",
            None,
        )
        st.warning(
            "هیچ‌کدام از نتایج ساختار قابل نمایش نداشتند."
        )

    with st.expander(
        "جزئیات فنی جست‌وجو"
    ):
        _render_telemetry(
            state[
                "telemetry"
            ]
        )
