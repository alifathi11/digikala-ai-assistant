
import json
from collections import Counter
from pathlib import Path
import time

import pandas as pd

from .analytics_metrics import (
    DEFAULT_WEIGHTS,
    comparison_fact_accuracy,
    expected_fact_values,
    fact_value_accuracy,
    policy_guard_configuration,
    rendered_metric_accuracy,
    scope_product_count_accuracy,
    summarize_analytics_results,
    validate_weights,
    weighted_judge_score,
)


class ManagerAnalyticsEvaluator:

    def __init__(
        self,
        analytics_pipeline,
        judge,
        dataset,
        generic_brand_values,
        rating_max=100.0,
        weights=None,
    ):
        self.pipeline = (
            analytics_pipeline
        )

        self.judge = judge
        self.dataset = dataset

        self.generic_brand_values = list(
            generic_brand_values
        )

        self.rating_max = float(
            rating_max
        )

        self.weights = dict(
            weights
            or DEFAULT_WEIGHTS
        )

        validate_weights(
            self.weights
        )


    @staticmethod
    def _load_checkpoint(
        path,
    ):
        path = Path(
            path
        )

        if not path.exists():
            return {}

        rows = {}

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(
                    line
                )

                if row.get(
                    "status"
                ) == "ok":
                    rows[
                        str(
                            row[
                                "case_id"
                            ]
                        )
                    ] = row

        return rows


    @staticmethod
    def _append_checkpoint(
        path,
        row,
    ):
        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


    @staticmethod
    def _judge_scores(
        payload,
    ):
        return {
            key: float(
                value[
                    "score"
                ]
            )
            for key, value
            in payload.items()
            if (
                isinstance(
                    value,
                    dict,
                )
                and "score"
                in value
            )
        }


    @staticmethod
    def _compact_result(
        result,
    ):
        return {
            "question": (
                result.get(
                    "question"
                )
            ),
            "answer": (
                result.get(
                    "answer"
                )
            ),
            "answer_template": (
                result.get(
                    "answer_template"
                )
            ),
            "insights": (
                result.get(
                    "insights"
                )
                or []
            ),
            "caveats": (
                result.get(
                    "caveats"
                )
                or []
            ),
            "confidence": (
                result.get(
                    "confidence"
                )
            ),
            "numeric_faithfulness_valid": (
                result.get(
                    "numeric_faithfulness_valid"
                )
            ),
            "validation_errors": (
                result.get(
                    "validation_errors"
                )
                or []
            ),
            "repaired": (
                result.get(
                    "repaired"
                )
            ),
            "facts": (
                result.get(
                    "facts"
                )
                or {}
            ),
            "context": {
                "scope": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "scope",
                        {},
                    )
                ),
                "data_quality": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "data_quality",
                        {},
                    )
                ),
                "top_brands": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "top_brands",
                        [],
                    )
                ),
                "top_products_by_rating": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "top_products_by_rating",
                        [],
                    )
                ),
                "top_products_by_rating_count": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "top_products_by_rating_count",
                        [],
                    )
                ),
                "category_comparison": (
                    result.get(
                        "context",
                        {}
                    ).get(
                        "category_comparison",
                        [],
                    )
                ),
            },
            "telemetry": (
                result.get(
                    "telemetry"
                )
                or {}
            ),
        }


    def _evaluate_one(
        self,
        case,
    ):
        start = time.perf_counter()

        result = (
            self.pipeline
            .answer(
                question=case[
                    "question"
                ],
                filters=case.get(
                    "filters"
                ),
                comparison_categories=(
                    case.get(
                        "comparison_categories"
                    )
                ),
                category_field=case.get(
                    "category_field",
                    "Category2",
                ),
            )
        )

        compact = (
            self._compact_result(
                result
            )
        )

        expected = expected_fact_values(
            product_frame=(
                self.pipeline
                .analytics
                .products
            ),
            case=case,
            generic_brand_values=(
                self.generic_brand_values
            ),
            rating_max=(
                self.rating_max
            ),
        )

        fact_check = fact_value_accuracy(
            generated_facts=(
                compact[
                    "facts"
                ]
            ),
            expected_values=(
                expected
            ),
        )

        scope_accuracy = (
            scope_product_count_accuracy(
                generated_facts=(
                    compact[
                        "facts"
                    ]
                ),
                expected_values=(
                    expected
                ),
            )
        )

        comparison_accuracy = (
            comparison_fact_accuracy(
                generated_facts=(
                    compact[
                        "facts"
                    ]
                ),
                expected_values=(
                    expected
                ),
            )
        )

        render_accuracy = (
            rendered_metric_accuracy(
                compact
            )
        )

        policy_guard = (
            policy_guard_configuration(
                result=compact,
                policy_expectations=(
                    case.get(
                        "policy_expectations"
                    )
                ),
            )
        )

        judge_result = (
            self.judge
            .judge(
                case=case,
                generated_result=(
                    compact
                ),
            )
        )

        judge_payload = (
            judge_result[
                "payload"
            ]
        )

        judge_scores = (
            self._judge_scores(
                judge_payload
            )
        )

        overall_score = (
            weighted_judge_score(
                scores=judge_scores,
                weights=(
                    self.weights
                ),
            )
        )

        answer_telemetry = (
            compact[
                "telemetry"
            ]
        )

        judge_telemetry = (
            judge_result[
                "telemetry"
            ]
        )

        total_latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "case_id": case[
                "case_id"
            ],
            "split": case[
                "split"
            ],
            "case_type": case[
                "case_type"
            ],
            "status": "ok",
            "question": case[
                "question"
            ],
            "filters": case.get(
                "filters"
            ),
            "comparison_categories": (
                case.get(
                    "comparison_categories"
                )
            ),
            "policy_expectations": (
                case.get(
                    "policy_expectations"
                )
            ),
            "answer": compact[
                "answer"
            ],
            "confidence": compact[
                "confidence"
            ],
            "repaired": bool(
                compact[
                    "repaired"
                ]
            ),
            "numeric_faithfulness": float(
                bool(
                    compact[
                        "numeric_faithfulness_valid"
                    ]
                )
            ),
            "fact_value_accuracy": (
                fact_check[
                    "accuracy"
                ]
            ),
            "fact_values_correct": (
                fact_check[
                    "correct"
                ]
            ),
            "fact_values_checked": (
                fact_check[
                    "checked"
                ]
            ),
            "fact_value_errors": (
                fact_check[
                    "errors"
                ]
            ),
            "scope_product_count_accuracy": (
                scope_accuracy
            ),
            "comparison_fact_accuracy": (
                comparison_accuracy
            ),
            "rendered_metric_accuracy": (
                render_accuracy
            ),
            "policy_guard_configuration": (
                policy_guard
            ),
            "overall_judge_score": (
                overall_score
            ),
            **judge_scores,
            "judge_failure_tags": (
                judge_payload[
                    "failure_tags"
                ]
            ),
            "judge_summary_reason": (
                judge_payload[
                    "summary_reason"
                ]
            ),
            "answer_latency_ms": float(
                answer_telemetry.get(
                    "total_latency_ms",
                    0.0,
                )
            ),
            "answer_generation_latency_ms": float(
                answer_telemetry.get(
                    "generation_latency_ms",
                    0.0,
                )
            ),
            "answer_total_tokens": int(
                answer_telemetry.get(
                    "total_tokens",
                    0,
                )
            ),
            "answer_cost_usd": (
                answer_telemetry.get(
                    "estimated_cost_usd"
                )
            ),
            "judge_latency_ms": float(
                judge_telemetry.get(
                    "latency_ms",
                    0.0,
                )
            ),
            "judge_total_tokens": int(
                judge_telemetry.get(
                    "total_tokens",
                    0,
                )
            ),
            "judge_cost_usd": (
                judge_telemetry.get(
                    "estimated_cost_usd"
                )
            ),
            "evaluation_latency_ms": float(
                total_latency_ms
            ),
            "result": compact,
            "judge": judge_payload,
        }


    def run(
        self,
        checkpoint_path,
        splits=None,
        force=False,
    ):
        checkpoint_path = Path(
            checkpoint_path
        )

        completed = (
            {}
            if force
            else self._load_checkpoint(
                checkpoint_path
            )
        )

        selected = [
            case
            for case in (
                self.dataset.cases
            )
            if (
                not splits
                or case[
                    "split"
                ]
                in set(
                    splits
                )
            )
        ]

        rows = []

        for index, case in enumerate(
            selected,
            start=1,
        ):
            case_id = case[
                "case_id"
            ]

            if (
                not force
                and case_id
                in completed
            ):
                rows.append(
                    completed[
                        case_id
                    ]
                )

                print(
                    f"[{index}/{len(selected)}] "
                    f"{case_id}: checkpoint"
                )

                continue

            print(
                f"[{index}/{len(selected)}] "
                f"{case_id}: running"
            )

            try:
                row = (
                    self._evaluate_one(
                        case
                    )
                )
            except Exception as error:
                row = {
                    "case_id": case_id,
                    "split": case[
                        "split"
                    ],
                    "case_type": case[
                        "case_type"
                    ],
                    "status": "error",
                    "question": case[
                        "question"
                    ],
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }

            self._append_checkpoint(
                checkpoint_path,
                row,
            )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )


def analytics_failure_summary(
    frame,
):
    counter = Counter()

    for tags in frame.get(
        "judge_failure_tags",
        [],
    ):
        if not isinstance(
            tags,
            list,
        ):
            continue

        for tag in tags:
            counter[
                str(
                    tag
                )
            ] += 1

    for row in frame.itertuples(
        index=False
    ):
        if getattr(
            row,
            "status",
            None,
        ) != "ok":
            counter[
                "execution_error"
            ] += 1
            continue

        if float(
            getattr(
                row,
                "numeric_faithfulness",
                0.0,
            )
        ) < 1.0:
            counter[
                "numeric_faithfulness_failure"
            ] += 1

        if float(
            getattr(
                row,
                "fact_value_accuracy",
                0.0,
            )
        ) < 1.0:
            counter[
                "aggregation_fact_mismatch"
            ] += 1

        comparison_accuracy = getattr(
            row,
            "comparison_fact_accuracy",
            None,
        )

        if (
            comparison_accuracy
            is not None
            and not pd.isna(
                comparison_accuracy
            )
            and float(
                comparison_accuracy
            )
            < 1.0
        ):
            counter[
                "comparison_fact_mismatch"
            ] += 1

    return pd.DataFrame(
        [
            {
                "failure": key,
                "count": int(
                    value
                ),
            }
            for key, value
            in counter.most_common()
        ]
    )


def build_analytics_report(
    frame,
):
    summary = (
        summarize_analytics_results(
            frame
        )
    )

    failures = (
        analytics_failure_summary(
            frame
        )
    )

    costs = {}

    for prefix in (
        "answer",
        "judge",
    ):
        column = (
            f"{prefix}_cost_usd"
        )

        if column in frame.columns:
            values = pd.to_numeric(
                frame[
                    column
                ],
                errors="coerce",
            )

            if values.notna().any():
                costs[
                    f"{prefix}_cost_usd"
                ] = float(
                    values.sum()
                )

    if costs:
        costs[
            "total_cost_usd"
        ] = float(
            sum(
                costs.values()
            )
        )

    return {
        "benchmark": (
            "Manager Analytics "
            "LLM-assisted evaluation"
        ),
        "case_count": int(
            len(
                frame
            )
        ),
        "successful_cases": int(
            (
                frame[
                    "status"
                ]
                == "ok"
            ).sum()
        ),
        "summary": (
            summary
            .where(
                pd.notna(
                    summary
                ),
                None,
            )
            .to_dict(
                orient="records"
            )
        ),
        "failures": (
            failures
            .to_dict(
                orient="records"
            )
        ),
        "cost": costs,
        "caveat": (
            "Qualitative answer-quality scores are produced by an "
            "LLM-assisted judge and are proxy evaluation, not independent "
            "human annotation. Numeric faithfulness and deterministic fact "
            "accuracy are checked independently in Python."
        ),
    }


def export_analytics_manual_review(
    frame,
    path,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        column
        for column in [
            "case_id",
            "split",
            "case_type",
            "question",
            "answer",
            "overall_judge_score",
            "numeric_faithfulness",
            "fact_value_accuracy",
            "comparison_fact_accuracy",
            "judge_failure_tags",
            "judge_summary_reason",
        ]
        if column in frame.columns
    ]

    output = frame[
        columns
    ].copy()

    output[
        "manual_status"
    ] = ""

    output[
        "manual_notes"
    ] = ""

    output.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
