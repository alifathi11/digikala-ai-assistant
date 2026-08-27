__all__ = [
    "GroundedQAPipeline",
    "ProductSearchPipeline",
    "ProductComparisonPipeline",
    "ManagerAnalyticsPipeline",
]


def __getattr__(name):
    if name == "GroundedQAPipeline":
        from .qa import GroundedQAPipeline
        return GroundedQAPipeline

    if name == "ProductSearchPipeline":
        from .product_search import ProductSearchPipeline
        return ProductSearchPipeline

    if name == "ProductComparisonPipeline":
        from .comparison import ProductComparisonPipeline
        return ProductComparisonPipeline

    if name == "ManagerAnalyticsPipeline":
        from .analytics import ManagerAnalyticsPipeline
        return ManagerAnalyticsPipeline

    raise AttributeError(name)
