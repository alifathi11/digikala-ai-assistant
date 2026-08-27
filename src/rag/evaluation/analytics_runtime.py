
from dataclasses import dataclass
from pathlib import Path

from ..analytics import (
    AnalyticsRepository,
    AnalyticsService,
)
from ..config import load_config
from ..generation import OpenAIJSONGenerator
from ..pipeline.analytics import (
    ManagerAnalyticsPipeline,
)
from .analytics_judge import (
    ManagerAnalyticsJudge,
)


@dataclass
class AnalyticsEvaluationContext:
    pipeline: object
    judge: object
    analytics_service: object
    evaluation_config: dict
    analytics_config: dict


def load_analytics_evaluation_context(
    project_root,
    api_key,
    base_url,
):
    project_root = Path(
        project_root
    )

    analytics_config = load_config(
        project_root
        / "configs"
        / "analytics.yaml"
    )[
        "analytics"
    ]

    qa_config = load_config(
        project_root
        / "configs"
        / "qa.yaml"
    )

    evaluation_config = load_config(
        project_root
        / "configs"
        / "analytics_evaluation.yaml"
    )[
        "analytics_evaluation"
    ]

    repository = (
        AnalyticsRepository
        .from_project_root(
            project_root
        )
    )

    audit_config = analytics_config[
        "audit"
    ]

    aggregation_config = analytics_config[
        "aggregation"
    ]

    manager_config = analytics_config[
        "manager_qa"
    ]

    service = AnalyticsService(
        repository=repository,
        generic_brand_values=(
            audit_config[
                "generic_brand_values"
            ]
        ),
        unknown_category_values=(
            audit_config[
                "unknown_category_values"
            ]
        ),
        min_rating_count_for_leaders=(
            aggregation_config[
                "min_rating_count_for_leaders"
            ]
        ),
        default_top_n=(
            aggregation_config[
                "default_top_n"
            ]
        ),
        product_rating_max=(
            aggregation_config[
                "product_rating_max"
            ]
        ),
    )

    generation_config = qa_config[
        "generation"
    ]

    answer_generator = (
        OpenAIJSONGenerator(
            api_key=api_key,
            base_url=base_url,
            model=generation_config[
                "model"
            ],
            input_cost_per_million=(
                generation_config.get(
                    "input_cost_per_million"
                )
            ),
            output_cost_per_million=(
                generation_config.get(
                    "output_cost_per_million"
                )
            ),
        )
    )

    pipeline = ManagerAnalyticsPipeline(
        analytics_service=service,
        generator=answer_generator,
        brand_usable_coverage=(
            manager_config[
                "brand_usable_coverage"
            ]
        ),
        product_rating_coverage=(
            manager_config[
                "product_rating_coverage"
            ]
        ),
        historical_price_enabled=(
            manager_config[
                "historical_price_enabled"
            ]
        ),
        review_volume_ranking_enabled=(
            manager_config[
                "review_volume_ranking_enabled"
            ]
        ),
        top_n=(
            manager_config[
                "top_n"
            ]
        ),
    )

    judge_config = evaluation_config[
        "judge"
    ]

    judge_generator = (
        OpenAIJSONGenerator(
            api_key=api_key,
            base_url=base_url,
            model=(
                judge_config.get(
                    "model"
                )
                or generation_config[
                    "model"
                ]
            ),
            input_cost_per_million=(
                judge_config.get(
                    "input_cost_per_million",
                    generation_config.get(
                        "input_cost_per_million"
                    ),
                )
            ),
            output_cost_per_million=(
                judge_config.get(
                    "output_cost_per_million",
                    generation_config.get(
                        "output_cost_per_million"
                    ),
                )
            ),
        )
    )

    judge = ManagerAnalyticsJudge(
        generator=judge_generator
    )

    return AnalyticsEvaluationContext(
        pipeline=pipeline,
        judge=judge,
        analytics_service=service,
        evaluation_config=(
            evaluation_config
        ),
        analytics_config=(
            analytics_config
        ),
    )
