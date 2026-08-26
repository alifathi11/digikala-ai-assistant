from src.rag.evaluation.product_search_qrels import (
    ProductSearchQrelsBuilder,
)


class FakeMetadata:
    @staticmethod
    def _process(text):
        return text


def _builder():
    builder = object.__new__(
        ProductSearchQrelsBuilder
    )
    builder.metadata = FakeMetadata()
    builder.rescue_max_fragments = 8
    return builder


def test_rescue_keeps_product_type_phrase():
    fragments = _builder()._rescue_fragments(
        "سرخ کن بدون روغن با ظرفیت حداقل 6 لیتر"
    )

    assert "سرخ کن" in fragments


def test_rescue_removes_numeric_noise():
    fragments = _builder()._rescue_fragments(
        "پاوربانک شیائومی 20000"
    )

    assert "پاوربانک شیائومی" in fragments

    assert all(
        "20000" not in fragment
        for fragment in fragments
    )
