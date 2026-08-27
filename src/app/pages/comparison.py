import html
import time
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.app.safety import (
    render_ui_error,
    require_mapping,
)


STANCE_LABELS = {
    "positive": "مثبت",
    "mixed": "ترکیبی",
    "negative": "منفی",
    "unknown": "شاهد ناکافی",
}

CONFIDENCE_LABELS = {
    "high": "بالا",
    "medium": "متوسط",
    "low": "پایین",
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


def _state():
    if (
        "comparison_selected_ids"
        not in st.session_state
    ):
        st.session_state[
            "comparison_selected_ids"
        ] = []

    return st.session_state[
        "comparison_selected_ids"
    ]


def _clear_result():
    st.session_state.pop(
        "comparison_result",
        None,
    )


def _add_product(
    product_id,
):
    selected = list(
        _state()
    )

    product_id = int(
        product_id
    )

    if product_id in selected:
        return

    if len(selected) >= 3:
        return

    selected.append(
        product_id
    )

    st.session_state[
        "comparison_selected_ids"
    ] = selected

    _clear_result()


def _remove_product(
    product_id,
):
    product_id = int(
        product_id
    )

    st.session_state[
        "comparison_selected_ids"
    ] = [
        value
        for value
        in _state()
        if int(value)
        != product_id
    ]

    _clear_result()


def _product_map(
    products,
):
    return {
        int(row.id): row
        for row
        in products.itertuples(
            index=False
        )
    }


def _evidence_map(
    evidence_documents,
):
    if (
        evidence_documents is None
        or len(
            evidence_documents
        )
        == 0
    ):
        return {}

    result = {}

    for row in (
        evidence_documents
        .itertuples(
            index=False
        )
    ):
        try:
            product_id = int(
                getattr(
                    row,
                    "product_id"
                )
            )

            comment_id = int(
                getattr(
                    row,
                    "id"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        result[
            (
                product_id,
                comment_id,
            )
        ] = row

    return result


def _render_product_summary_card(
    row,
    selected=False,
):
    product_id = int(
        _safe_value(
            row,
            "id",
            -1,
        )
    )

    title = _safe_value(
        row,
        "title_fa",
        "بدون عنوان",
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

    rate = _safe_value(
        row,
        "Rate",
    )

    price = _format_price(
        _safe_value(
            row,
            "Price",
            None,
        )
    )

    state_class = (
        " comparison-product-selected"
        if selected
        else ""
    )

    card_html = dedent(
        f"""
        <div class="comparison-product-card{state_class}">
            <div class="comparison-product-id">#{product_id}</div>
            <div class="comparison-product-title">{_escape(title)}</div>
            <div class="product-meta comparison-product-meta">
                <span class="meta-chip"><strong>برند</strong> {_escape(brand)}</span>
                <span class="meta-chip"><strong>دسته</strong> {_escape(category)}</span>
                <span class="meta-chip"><strong>امتیاز</strong> {_escape(rate)}</span>
                <span class="meta-chip"><strong>قیمت</strong> {_escape(price)}</span>
            </div>
        </div>
        """
    ).strip()

    st.html(
        card_html
    )


def _render_search_results(
    services,
):
    results = st.session_state.get(
        "comparison_search_results"
    )

    if (
        results is None
        or len(results) == 0
    ):
        return

    selected = set(
        int(value)
        for value
        in _state()
    )

    st.html(
        dedent(
            f"""
            <div class="comparison-section-head">
                <div>
                    <span>نتایج جست‌وجوی محصول</span>
                    <strong>{len(results)} گزینه</strong>
                </div>
            </div>
            """
        ).strip()
    )

    for row in results.itertuples(
        index=False
    ):
        product_id = int(
            getattr(
                row,
                "id"
            )
        )

        _render_product_summary_card(
            row,
            selected=(
                product_id
                in selected
            ),
        )

        if product_id in selected:
            st.button(
                "✓ به مقایسه اضافه شده",
                key=(
                    "comparison_added_"
                    f"{product_id}"
                ),
                use_container_width=True,
                disabled=True,
            )
        else:
            st.button(
                "افزودن به مقایسه",
                key=(
                    "comparison_add_"
                    f"{product_id}"
                ),
                use_container_width=True,
                disabled=(
                    len(selected)
                    >= 3
                ),
                on_click=(
                    _add_product
                ),
                args=(
                    product_id,
                ),
            )


def _selected_products(
    services,
):
    selected_ids = list(
        _state()
    )

    if not selected_ids:
        return (
            services
            .product_search
            .products
            .iloc[
                0:0
            ]
            .copy()
        )

    return (
        services
        .comparison
        .context_service
        .get_products(
            selected_ids
        )
    )


def _render_selected_products(
    services,
):
    products = _selected_products(
        services
    )

    selected_ids = list(
        _state()
    )

    st.html(
        dedent(
            f"""
            <div class="comparison-section-head comparison-selected-head">
                <div>
                    <span>محصولات انتخاب‌شده</span>
                    <strong>{len(selected_ids)} از 3</strong>
                </div>
                <div class="comparison-selection-hint">
                    برای مقایسه حداقل دو محصول انتخاب کنید.
                </div>
            </div>
            """
        ).strip()
    )

    if len(products) == 0:
        st.html(
            """
            <div class="comparison-empty-state">
                هنوز محصولی برای مقایسه انتخاب نشده است.
            </div>
            """
        )
        return

    columns = st.columns(
        len(products)
    )

    for column, row in zip(
        columns,
        products.itertuples(
            index=False
        ),
    ):
        product_id = int(
            getattr(
                row,
                "id"
            )
        )

        with column:
            _render_product_summary_card(
                row,
                selected=True,
            )

            st.button(
                "حذف از مقایسه",
                key=(
                    "comparison_remove_"
                    f"{product_id}"
                ),
                use_container_width=True,
                on_click=(
                    _remove_product
                ),
                args=(
                    product_id,
                ),
            )


def _render_evidence(
    assessment,
    evidence_lookup,
):
    evidence_ids = assessment.get(
        "evidence_ids",
        [],
    )

    product_id = int(
        assessment[
            "product_id"
        ]
    )

    if not evidence_ids:
        st.caption(
            "شاهد مستقیمی برای این ارزیابی "
            "انتخاب نشده است."
        )
        return

    rendered = 0

    for evidence_id in evidence_ids:
        try:
            evidence_id = int(
                evidence_id
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        row = evidence_lookup.get(
            (
                product_id,
                evidence_id,
            )
        )

        if row is None:
            continue

        rate = _safe_value(
            row,
            "rate",
        )

        body = _safe_value(
            row,
            "body",
            "",
        )

        st.html(
            dedent(
                f"""
                <div class="evidence-card comparison-evidence-card">
                    <div class="evidence-meta">
                        <span>نظر #{evidence_id}</span>
                        <span>·</span>
                        <span>امتیاز {_escape(rate)}</span>
                    </div>
                    <div class="evidence-body">{_escape(body)}</div>
                </div>
                """
            ).strip()
        )

        rendered += 1

    if rendered == 0:
        st.caption(
            "شاهد استنادی قابل نمایش پیدا نشد."
        )


def _render_assessment(
    assessment,
    product_row,
    evidence_lookup,
):
    product_id = int(
        assessment[
            "product_id"
        ]
    )

    title = _safe_value(
        product_row,
        "title_fa",
        f"محصول {product_id}",
    )

    stance = str(
        assessment.get(
            "stance",
            "unknown",
        )
    ).strip().lower()

    stance_label = (
        STANCE_LABELS.get(
            stance,
            stance,
        )
    )

    assessment_text = assessment.get(
        "text",
        "",
    )

    evidence_ids = assessment.get(
        "evidence_ids",
        [],
    )

    st.html(
        dedent(
            f"""
            <div class="comparison-assessment comparison-stance-{_escape(stance)}">
                <div class="comparison-assessment-head">
                    <div>
                        <span class="comparison-assessment-id">#{product_id}</span>
                        <strong>{_escape(title)}</strong>
                    </div>
                    <span class="comparison-stance">{_escape(stance_label)}</span>
                </div>
                <div class="comparison-assessment-text">
                    {_escape(assessment_text)}
                </div>
                <div class="comparison-evidence-count">
                    {len(evidence_ids)} شاهد استنادی
                </div>
            </div>
            """
        ).strip()
    )

    with st.expander(
        (
            "مشاهده شواهد"
            f" · {len(evidence_ids)}"
        ),
        expanded=False,
    ):
        _render_evidence(
            assessment=assessment,
            evidence_lookup=(
                evidence_lookup
            ),
        )


def _render_result(
    result,
):
    products = result[
        "product_metadata"
    ]

    product_lookup = _product_map(
        products
    )

    evidence_lookup = (
        _evidence_map(
            result[
                "evidence_documents"
            ]
        )
    )

    overall_winner = result.get(
        "overall_winner_product_id"
    )

    winner_title = "بدون برنده قطعی"

    if overall_winner is not None:
        try:
            overall_winner = int(
                overall_winner
            )
        except (
            TypeError,
            ValueError,
        ):
            overall_winner = None

    if (
        overall_winner is not None
        and overall_winner
        in product_lookup
    ):
        winner_title = _safe_value(
            product_lookup[
                overall_winner
            ],
            "title_fa",
            f"محصول {overall_winner}",
        )

    confidence = str(
        result.get(
            "confidence",
            "low",
        )
    ).lower()

    confidence_label = (
        CONFIDENCE_LABELS.get(
            confidence,
            confidence,
        )
    )

    citation_label = (
        "معتبر"
        if result.get(
            "citation_valid"
        )
        else "نیازمند بررسی"
    )

    st.html(
        dedent(
            f"""
            <div class="comparison-result-hero">
                <div class="comparison-result-copy">
                    <span class="comparison-result-kicker">نتیجه مقایسه</span>
                    <h2>{_escape(result.get("summary", ""))}</h2>
                    <p>{_escape(result.get("overall_recommendation", ""))}</p>
                </div>
                <div class="comparison-result-winner">
                    <span>انتخاب کلی</span>
                    <strong>{_escape(winner_title)}</strong>
                    <small>اعتماد: {_escape(confidence_label)}</small>
                </div>
            </div>
            """
        ).strip()
    )

    stat_columns = st.columns(
        3
    )

    stat_columns[
        0
    ].metric(
        "معیارها",
        len(
            result.get(
                "criteria",
                [],
            )
        ),
    )

    stat_columns[
        1
    ].metric(
        "اعتبار استناد",
        citation_label,
    )

    stat_columns[
        2
    ].metric(
        "شواهد استفاده‌شده",
        sum(
            len(
                values
            )
            for values
            in result.get(
                "evidence_ids_by_product",
                {},
            ).values()
        ),
    )

    for criterion_index, criterion in enumerate(
        result.get(
            "criteria",
            [],
        ),
        start=1,
    ):
        criterion_name = criterion.get(
            "name",
            f"معیار {criterion_index}",
        )

        winner_product_id = (
            criterion.get(
                "winner_product_id"
            )
        )

        winner_text = (
            "برنده قطعی ندارد"
        )

        if winner_product_id is not None:
            try:
                winner_product_id = int(
                    winner_product_id
                )
            except (
                TypeError,
                ValueError,
            ):
                winner_product_id = None

        if (
            winner_product_id is not None
            and winner_product_id
            in product_lookup
        ):
            winner_text = _safe_value(
                product_lookup[
                    winner_product_id
                ],
                "title_fa",
                f"محصول {winner_product_id}",
            )

        winner_reason = criterion.get(
            "winner_reason",
            "",
        )

        st.html(
            dedent(
                f"""
                <div class="comparison-criterion-head">
                    <div class="comparison-criterion-index">{criterion_index}</div>
                    <div>
                        <span>معیار مقایسه</span>
                        <h3>{_escape(criterion_name)}</h3>
                    </div>
                    <div class="comparison-criterion-winner">
                        <span>نتیجه این معیار</span>
                        <strong>{_escape(winner_text)}</strong>
                    </div>
                </div>
                """
            ).strip()
        )

        if winner_reason:
            st.html(
                dedent(
                    f"""
                    <div class="comparison-winner-reason">
                        {_escape(winner_reason)}
                    </div>
                    """
                ).strip()
            )

        for assessment in criterion.get(
            "assessments",
            [],
        ):
            try:
                product_id = int(
                    assessment[
                        "product_id"
                    ]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            product_row = (
                product_lookup.get(
                    product_id
                )
            )

            if product_row is None:
                continue

            _render_assessment(
                assessment=assessment,
                product_row=(
                    product_row
                ),
                evidence_lookup=(
                    evidence_lookup
                ),
            )

    telemetry = result.get(
        "telemetry",
        {},
    )

    with st.expander(
        "جزئیات فنی مقایسه",
        expanded=False,
    ):
        columns = st.columns(
            4
        )

        total_latency_ms = telemetry.get(
            "total_latency_ms"
        )

        generation_latency_ms = telemetry.get(
            "generation_latency_ms"
        )

        review_latency_ms = telemetry.get(
            "review_retrieval_latency_ms"
        )

        columns[
            0
        ].metric(
            "زمان کل",
            (
                f"{float(total_latency_ms) / 1000:.2f}s"
                if total_latency_ms
                is not None
                else "—"
            ),
        )

        columns[
            1
        ].metric(
            "LLM",
            (
                f"{float(generation_latency_ms) / 1000:.2f}s"
                if generation_latency_ms
                is not None
                else "—"
            ),
        )

        columns[
            2
        ].metric(
            "بازیابی نظرات",
            (
                f"{float(review_latency_ms):.0f}ms"
                if review_latency_ms
                is not None
                else "—"
            ),
        )

        columns[
            3
        ].metric(
            "توکن",
            telemetry.get(
                "total_tokens",
                "—",
            ),
        )

        st.caption(
            (
                f"مدل: {telemetry.get('model', '—')} · "
                f"تعداد نظر بازیابی‌شده: "
                f"{telemetry.get('retrieved_review_count', '—')} · "
                f"citation repair: "
                f"{'بله' if result.get('citation_repaired') else 'خیر'}"
            )
        )


def render(
    services,
):
    st.html(
        dedent(
            """
            <div class="hero comparison-hero">
                <div class="hero-copy">
                    <div class="hero-kicker">● مقایسه‌ی مبتنی بر شواهد</div>
                    <h1>محصول‌ها را کنار هم، با تکیه بر تجربه‌ی کاربران مقایسه کنید</h1>
                    <p>
                        دو یا سه محصول را پیدا کنید، معیار موردنظرتان را بنویسید
                        و تفاوت‌ها را با استناد به نظرات واقعی کاربران ببینید.
                    </p>
                </div>
                <div class="hero-icon">⚖</div>
            </div>
            """
        ).strip()
    )

    selected_ids = list(
        _state()
    )

    with st.form(
        "comparison_product_search_form",
        clear_on_submit=False,
    ):
        search_query = st.text_input(
            "جست‌وجوی محصول",
            placeholder=(
                "مثلاً ضد آفتاب ژیناژن، "
                "شامپو سریتا یا پاوربانک شیائومی"
            ),
        )

        submitted = (
            st.form_submit_button(
                "پیدا کردن محصول",
                use_container_width=True,
            )
        )

    if submitted:
        normalized = str(
            search_query
        ).strip()

        if not normalized:
            st.warning(
                "عبارت جست‌وجوی محصول را وارد کنید."
            )
        else:
            try:
                with st.spinner(
                    "در حال جست‌وجوی محصول..."
                ):
                    results = (
                        services
                        .product_search
                        .metadata_retriever
                        .retrieve(
                            normalized,
                            top_k=8,
                        )
                    )

                if not isinstance(
                    results,
                    pd.DataFrame,
                ):
                    raise ValueError(
                        "خروجی جست‌وجوی محصول ساختار جدولی ندارد."
                    )

                st.session_state[
                    "comparison_search_query"
                ] = normalized

                st.session_state[
                    "comparison_search_results"
                ] = results.copy()

            except Exception as exc:
                st.session_state.pop(
                    "comparison_search_results",
                    None,
                )
                render_ui_error(
                    "جست‌وجوی محصول برای مقایسه قابل پردازش نبود.",
                    exc,
                )

    _render_search_results(
        services
    )

    st.divider()

    _render_selected_products(
        services
    )

    selected_ids = list(
        _state()
    )

    if len(selected_ids) >= 2:
        st.html(
            """
            <div class="comparison-query-intro">
                حالا بگویید دقیقاً از چه نظر می‌خواهید این محصولات مقایسه شوند.
            </div>
            """
        )

        with st.form(
            "comparison_run_form",
            clear_on_submit=False,
        ):
            comparison_query = (
                st.text_area(
                    "معیار مقایسه",
                    value=(
                        "این محصولات را از نظر تفاوت‌های مهم، "
                        "نقاط قوت و ضعف و تجربه‌ی کاربران مقایسه کن "
                        "و اگر شواهد کافی است مناسب‌ترین گزینه را مشخص کن."
                    ),
                    height=110,
                )
            )

            compare_submitted = (
                st.form_submit_button(
                    "مقایسه‌ی هوشمند",
                    use_container_width=True,
                )
            )

        if compare_submitted:
            normalized_query = str(
                comparison_query
            ).strip()

            if not normalized_query:
                st.warning(
                    "معیار مقایسه را وارد کنید."
                )
            else:
                try:
                    with st.spinner(
                        "در حال بررسی شواهد و مقایسه‌ی محصولات..."
                    ):
                        start = time.perf_counter()

                        result = (
                            services
                            .comparison
                            .compare(
                                product_ids=(
                                    selected_ids
                                ),
                                query=(
                                    normalized_query
                                ),
                            )
                        )

                    result = require_mapping(
                        result,
                        required_keys=(
                            "product_ids",
                        ),
                        label="خروجی مقایسه",
                    )

                    result_product_ids = result.get(
                        "product_ids"
                    )

                    if not isinstance(
                        result_product_ids,
                        (
                            list,
                            tuple,
                        ),
                    ):
                        raise ValueError(
                            "فهرست شناسه‌های خروجی مقایسه معتبر نیست."
                        )

                    telemetry = result.get(
                        "telemetry"
                    )

                    if not isinstance(
                        telemetry,
                        dict,
                    ):
                        telemetry = {}
                        result[
                            "telemetry"
                        ] = telemetry

                    telemetry[
                        "ui_total_latency_ms"
                    ] = (
                        time.perf_counter()
                        - start
                    ) * 1000

                    st.session_state[
                        "comparison_result"
                    ] = result

                except Exception as exc:
                    st.session_state.pop(
                        "comparison_result",
                        None,
                    )
                    render_ui_error(
                        "مدل نتیجه‌ی مقایسه‌ی قابل نمایش برنگرداند.",
                        exc,
                    )

    else:
        st.info(
            "برای شروع مقایسه حداقل دو محصول انتخاب کنید."
        )

    result = st.session_state.get(
        "comparison_result"
    )

    if result is not None:
        current_ids = [
            int(value)
            for value
            in selected_ids
        ]

        result_ids = [
            int(value)
            for value
            in result.get(
                "product_ids",
                [],
            )
        ]

        if current_ids == result_ids:
            st.divider()

            try:
                _render_result(
                    result
                )
            except Exception as exc:
                st.session_state.pop(
                    "comparison_result",
                    None,
                )
                render_ui_error(
                    "نتیجه‌ی مقایسه تولید شد اما ساختار آن قابل نمایش نبود.",
                    exc,
                )
