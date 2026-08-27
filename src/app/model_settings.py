from copy import copy

import streamlit as st

from src.rag.generation import (
    OpenAIJSONGenerator,
)


SUPPORTED_MODELS = (
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6-luna",
)

SECTION_LABELS = {
    "qa": "Grounded QA",
    "product_search": "Product Search",
    "comparison": "Product Comparison",
    "analytics": "Manager Analytics",
}

DEFAULT_MODELS = {
    "qa": "gpt-5.6-terra",
    "product_search": "gpt-5.6-terra",
    "comparison": "gpt-5.6-terra",
    "analytics": "gpt-5.6-terra",
}

_PENDING_KEY = (
    "_ui_model_pending"
)

_APPLIED_KEY = (
    "_ui_model_applied"
)

_SESSION_SERVICES_KEY = (
    "_ui_session_services"
)


def _initial_models():
    return dict(
        DEFAULT_MODELS
    )


def _ensure_state():
    if _PENDING_KEY not in (
        st.session_state
    ):
        st.session_state[
            _PENDING_KEY
        ] = _initial_models()

    if _APPLIED_KEY not in (
        st.session_state
    ):
        st.session_state[
            _APPLIED_KEY
        ] = {}


def render_model_settings():
    """
    Render model selectors in the current Streamlit container.

    Returns:
        (selected_models, apply_requested)
    """
    _ensure_state()

    pending = dict(
        st.session_state[
            _PENDING_KEY
        ]
    )

    applied = dict(
        st.session_state[
            _APPLIED_KEY
        ]
    )

    with st.expander(
        "⚙️ تنظیم مدل‌ها",
        expanded=False,
    ):
        st.caption(
            "مدل هر قابلیت را مستقل انتخاب کنید."
        )

        for section in (
            "qa",
            "product_search",
            "comparison",
            "analytics",
        ):
            current = pending.get(
                section,
                DEFAULT_MODELS[
                    section
                ],
            )

            try:
                index = (
                    SUPPORTED_MODELS
                    .index(
                        current
                    )
                )
            except ValueError:
                index = 0

            selected = st.selectbox(
                SECTION_LABELS[
                    section
                ],
                options=(
                    SUPPORTED_MODELS
                ),
                index=index,
                key=(
                    f"ui_model_select_"
                    f"{section}"
                ),
            )

            pending[
                section
            ] = selected

            active = applied.get(
                section
            )

            if active:
                st.markdown(
                    (
                        "<div class="
                        '"model-setting-active">'
                        "فعال: "
                        f"<strong>{active}</strong>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.session_state[
            _PENDING_KEY
        ] = pending

        apply_requested = (
            st.button(
                "اعمال مدل‌ها",
                key=(
                    "apply_ui_models"
                ),
                use_container_width=True,
                type="primary",
            )
        )

        st.caption(
            "اگر endpoint یک مدل را پشتیبانی نکند، "
            "فقط همان درخواست خطا می‌دهد و UI فعال می‌ماند."
        )

    return (
        dict(
            pending
        ),
        bool(
            apply_requested
        ),
    )


def _clone_services(
    base_services,
):
    """
    Create session-local pipeline shells while sharing heavy retrieval/data.

    This prevents model selection in one Streamlit session from mutating the
    cached service object used by another session.
    """
    services = copy(
        base_services
    )

    services.qa = copy(
        base_services.qa
    )

    services.product_search = copy(
        base_services.product_search
    )

    if (
        getattr(
            base_services.product_search,
            "reranker",
            None,
        )
        is not None
    ):
        services.product_search.reranker = copy(
            base_services
            .product_search
            .reranker
        )

    services.comparison = copy(
        base_services.comparison
    )

    services.analytics = copy(
        base_services.analytics
    )

    return services


def get_session_services(
    base_services,
):
    if _SESSION_SERVICES_KEY not in (
        st.session_state
    ):
        st.session_state[
            _SESSION_SERVICES_KEY
        ] = _clone_services(
            base_services
        )

    return st.session_state[
        _SESSION_SERVICES_KEY
    ]


def _pricing_for_model(
    project_config,
    model,
):
    models = (
        project_config.get(
            "models",
            {}
        )
        if isinstance(
            project_config,
            dict,
        )
        else {}
    )

    pricing = (
        models.get(
            model,
            {}
        )
        or {}
    )

    return (
        pricing.get(
            "input_cost_per_million"
        ),
        pricing.get(
            "output_cost_per_million"
        ),
    )


def _make_generator(
    *,
    api_key,
    base_url,
    model,
    project_config,
):
    (
        input_cost,
        output_cost,
    ) = _pricing_for_model(
        project_config=(
            project_config
        ),
        model=model,
    )

    return OpenAIJSONGenerator(
        api_key=api_key,
        base_url=base_url,
        model=model,
        input_cost_per_million=(
            input_cost
        ),
        output_cost_per_million=(
            output_cost
        ),
    )


def apply_model_selection(
    *,
    services,
    selected_models,
    api_key,
    base_url,
    project_config,
):
    """
    Assign one independent generator to each UI feature.

    Heavy retrieval/index objects remain untouched.
    """
    selected = {
        section: str(
            selected_models[
                section
            ]
        )
        for section in (
            "qa",
            "product_search",
            "comparison",
            "analytics",
        )
    }

    invalid = [
        model
        for model in (
            selected.values()
        )
        if model not in (
            SUPPORTED_MODELS
        )
    ]

    if invalid:
        raise ValueError(
            "Unsupported UI model: "
            + ", ".join(
                sorted(
                    set(
                        invalid
                    )
                )
            )
        )

    qa_generator = _make_generator(
        api_key=api_key,
        base_url=base_url,
        model=selected[
            "qa"
        ],
        project_config=(
            project_config
        ),
    )

    search_generator = _make_generator(
        api_key=api_key,
        base_url=base_url,
        model=selected[
            "product_search"
        ],
        project_config=(
            project_config
        ),
    )

    comparison_generator = _make_generator(
        api_key=api_key,
        base_url=base_url,
        model=selected[
            "comparison"
        ],
        project_config=(
            project_config
        ),
    )

    analytics_generator = _make_generator(
        api_key=api_key,
        base_url=base_url,
        model=selected[
            "analytics"
        ],
        project_config=(
            project_config
        ),
    )

    services.qa.generator = (
        qa_generator
    )

    reranker = getattr(
        services.product_search,
        "reranker",
        None,
    )

    if reranker is not None:
        reranker.generator = (
            search_generator
        )

    services.comparison.generator = (
        comparison_generator
    )

    services.analytics.generator = (
        analytics_generator
    )

    st.session_state[
        _APPLIED_KEY
    ] = dict(
        selected
    )

    return dict(
        selected
    )


def ensure_models_applied(
    *,
    services,
    api_key,
    base_url,
    project_config,
):
    """
    Apply defaults once for a new session.
    """
    _ensure_state()

    applied = st.session_state[
        _APPLIED_KEY
    ]

    if applied:
        return dict(
            applied
        )

    pending = st.session_state[
        _PENDING_KEY
    ]

    return apply_model_selection(
        services=services,
        selected_models=pending,
        api_key=api_key,
        base_url=base_url,
        project_config=(
            project_config
        ),
    )


def get_applied_models():
    _ensure_state()

    return dict(
        st.session_state[
            _APPLIED_KEY
        ]
    )
