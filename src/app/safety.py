import streamlit as st


def render_ui_error(
    message,
    error=None,
    *,
    retry_hint=True,
):
    """Render a user-safe UI error without exposing a traceback."""
    st.error(
        str(message)
    )

    if retry_hint:
        st.caption(
            "می‌توانید دوباره تلاش کنید یا ورودی را تغییر دهید."
        )

    if error is not None:
        detail = (
            f"{type(error).__name__}: {error}"
        )

        with st.expander(
            "جزئیات خطا",
            expanded=False,
        ):
            st.code(
                detail
            )


def require_mapping(
    value,
    *,
    required_keys=(),
    label="پاسخ مدل",
):
    """Validate only the minimum structure needed by the UI."""
    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{label} ساختار قابل نمایش ندارد."
        )

    missing = [
        key
        for key in required_keys
        if key not in value
    ]

    if missing:
        raise ValueError(
            f"{label} فیلدهای لازم را ندارد: "
            + ", ".join(
                missing
            )
        )

    return value
