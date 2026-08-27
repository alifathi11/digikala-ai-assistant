import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .comparison_metrics import (
    DEFAULT_WEIGHTS,
    assessment_product_coverage,
    citation_ownership_rate,
    deterministic_winner_accuracy,
    expected_metadata_winner,
    no_winner_accuracy,
    summarize_comparison_results,
    validate_weights,
    weighted_judge_score,
)


class ProductComparisonEvaluator:

    def __init__(
        self,
        comparison_pipeline,
        judge,
        dataset,
        weights=None,
    ):
        self.comparison_pipeline = (
            comparison_pipeline
        )
        self.judge = judge
        self.dataset = dataset
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
    def _jsonable_product_metadata(
        frame,
    ):
        columns = [
            column
            for column in [
                "id",
                "title_fa",
                "Brand",
                "Category1",
                "Category2",
                "sub_category",
                "Price",
                "min_price_last_month",
                "Rate",
                "Rate_cnt",
            ]
            if column in frame.columns
        ]

        return (
            frame[
                columns
            ]
            .where(
                pd.notna(
                    frame[
                        columns
                    ]
                ),
                None,
            )
            .to_dict(
                orient="records"
            )
        )


    @staticmethod
    def _judge_scores(
        payload,
    ):
        return {
            dimension: float(
                value[
                    "score"
                ]
            )
            for dimension, value
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


    def _evaluate_one(
        self,
        case,
    ):
        start = time.perf_counter()

        result = (
            self.comparison_pipeline
            .compare(
                product_ids=(
                    case[
                        "product_ids"
                    ]
                ),
                query=case[
                    "query"
                ],
            )
        )

        judge_result = self.judge.judge(
            query=case[
                "query"
            ],
            product_metadata=(
                result[
                    "product_metadata"
                ]
            ),
            retrieved_reviews=(
                result[
                    "retrieved_reviews"
                ]
            ),
            generated_result=result,
            case_notes=case.get(
                "notes",
                "",
            ),
        )

        judge_payload = (
            judge_result[
                "payload"
            ]
        )

        expectation = expected_metadata_winner(
            product_metadata=(
                result[
                    "product_metadata"
                ]
            ),
            winner_rule=case.get(
                "winner_rule"
            ),
        )

        winner_accuracy = (
            deterministic_winner_accuracy(
                result=result,
                winner_expectation=(
                    expectation
                ),
            )
        )

        no_winner = no_winner_accuracy(
            result=result,
            expect_no_winner=case.get(
                "expect_no_winner",
                False,
            ),
        )

        deterministic_tags = []

        if not result.get(
            "citation_valid",
            False,
        ):
            deterministic_tags.append(
                "invalid_citation"
            )

        if result.get(
            "citation_repaired",
            False,
        ):
            deterministic_tags.append(
                "citation_repair_needed"
            )

        if (
            not math_is_nan(
                winner_accuracy
            )
            and winner_accuracy
            < 1.0
        ):
            deterministic_tags.append(
                "deterministic_winner_mismatch"
            )

        if (
            not math_is_nan(
                no_winner
            )
            and no_winner
            < 1.0
        ):
            deterministic_tags.append(
                "expected_no_winner_mismatch"
            )

        failure_tags = []

        for tag in (
            list(
                judge_payload[
                    "failure_tags"
                ]
            )
            + deterministic_tags
        ):
            if tag not in failure_tags:
                failure_tags.append(
                    tag
                )

        comparison_tel = result[
            "telemetry"
        ]
        judge_tel = judge_result[
            "telemetry"
        ]

        judge_scores = self._judge_scores(
            judge_payload
        )

        cited_count = int(
            sum(
                len(
                    values
                )
                for values
                in result.get(
                    "evidence_ids_by_product",
                    {},
                ).values()
            )
        )

        retrieved_count = int(
            comparison_tel.get(
                "retrieved_review_count",
                len(
                    result[
                        "retrieved_reviews"
                    ]
                ),
            )
        )

        row = {
            "status": "ok",
            "case_id": str(
                case[
                    "case_id"
                ]
            ),
            "split": str(
                case[
                    "split"
                ]
            ),
            "case_type": str(
                case[
                    "case_type"
                ]
            ),
            "query": str(
                case[
                    "query"
                ]
            ),
            "product_ids": [
                int(value)
                for value
                in case[
                    "product_ids"
                ]
            ],
            "product_metadata": (
                self._jsonable_product_metadata(
                    result[
                        "product_metadata"
                    ]
                )
            ),
            "summary": result.get(
                "summary",
                "",
            ),
            "criteria": result.get(
                "criteria",
                [],
            ),
            "overall_winner_product_id": (
                result.get(
                    "overall_winner_product_id"
                )
            ),
            "overall_recommendation": (
                result.get(
                    "overall_recommendation",
                    "",
                )
            ),
            "confidence": result.get(
                "confidence"
            ),
            "insufficient_evidence": (
                result.get(
                    "insufficient_evidence"
                )
            ),
            "overall_score": (
                weighted_judge_score(
                    judge_payload,
                    weights=self.weights,
                )
            ),
            **judge_scores,
            "citation_validity": float(
                bool(
                    result.get(
                        "citation_valid"
                    )
                )
            ),
            "citation_ownership_rate": (
                citation_ownership_rate(
                    result
                )
            ),
            "assessment_product_coverage": (
                assessment_product_coverage(
                    result
                )
            ),
            "citation_repaired": bool(
                result.get(
                    "citation_repaired"
                )
            ),
            "citation_retry_count": int(
                result.get(
                    "citation_retry_count",
                    0,
                )
            ),
            "retrieved_review_count": (
                retrieved_count
            ),
            "cited_review_count": cited_count,
            "winner_rule": case.get(
                "winner_rule"
            ),
            "deterministic_winner_available": bool(
                expectation.get(
                    "available",
                    False,
                )
            ),
            "deterministic_expected_winner": (
                expectation.get(
                    "expected_product_id"
                )
            ),
            "deterministic_winner_accuracy": (
                winner_accuracy
            ),
            "expect_no_winner": bool(
                case.get(
                    "expect_no_winner",
                    False,
                )
            ),
            "no_winner_accuracy": no_winner,
            "failure_tags": failure_tags,
            "judge_summary_reason": (
                judge_payload[
                    "summary_reason"
                ]
            ),
            "comparison_review_retrieval_latency_ms": float(
                comparison_tel.get(
                    "review_retrieval_latency_ms",
                    0.0,
                )
            ),
            "comparison_generation_latency_ms": float(
                comparison_tel.get(
                    "generation_latency_ms",
                    0.0,
                )
            ),
            "comparison_total_latency_ms": float(
                comparison_tel.get(
                    "total_latency_ms",
                    0.0,
                )
            ),
            "comparison_prompt_tokens": int(
                comparison_tel.get(
                    "prompt_tokens",
                    0,
                )
            ),
            "comparison_completion_tokens": int(
                comparison_tel.get(
                    "completion_tokens",
                    0,
                )
            ),
            "comparison_total_tokens": int(
                comparison_tel.get(
                    "total_tokens",
                    0,
                )
            ),
            "comparison_cost_usd": (
                comparison_tel.get(
                    "estimated_cost_usd"
                )
            ),
            "judge_latency_ms": float(
                judge_tel.get(
                    "latency_ms",
                    0.0,
                )
            ),
            "judge_prompt_tokens": int(
                judge_tel.get(
                    "prompt_tokens",
                    0,
                )
            ),
            "judge_completion_tokens": int(
                judge_tel.get(
                    "completion_tokens",
                    0,
                )
            ),
            "judge_total_tokens": int(
                judge_tel.get(
                    "total_tokens",
                    0,
                )
            ),
            "judge_cost_usd": judge_tel.get(
                "estimated_cost_usd"
            ),
            "end_to_end_latency_ms": float(
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            ),
            "end_to_end_tokens": int(
                comparison_tel.get(
                    "total_tokens",
                    0,
                )
                + judge_tel.get(
                    "total_tokens",
                    0,
                )
            ),
        }

        comparison_cost = row[
            "comparison_cost_usd"
        ]
        judge_cost = row[
            "judge_cost_usd"
        ]

        row[
            "end_to_end_cost_usd"
        ] = (
            float(
                comparison_cost
            )
            + float(
                judge_cost
            )
            if (
                comparison_cost
                is not None
                and judge_cost
                is not None
            )
            else None
        )

        return row


    def evaluate(
        self,
        checkpoint_path,
        resume=True,
        splits=None,
        max_cases=None,
    ):
        checkpoint_path = Path(
            checkpoint_path
        )

        completed = (
            self._load_checkpoint(
                checkpoint_path
            )
            if resume
            else {}
        )

        allowed_splits = (
            {
                str(value)
                for value
                in splits
            }
            if splits
            else None
        )

        cases = [
            case
            for case
            in self.dataset.cases
            if (
                allowed_splits is None
                or case[
                    "split"
                ]
                in allowed_splits
            )
        ]

        if max_cases is not None:
            cases = cases[
                :int(
                    max_cases
                )
            ]

        rows = []

        for case in cases:
            case_id = str(
                case[
                    "case_id"
                ]
            )

            if (
                resume
                and case_id
                in completed
            ):
                rows.append(
                    completed[
                        case_id
                    ]
                )
                continue

            try:
                row = self._evaluate_one(
                    case
                )
            except Exception as error:
                row = {
                    "status": "error",
                    "case_id": case_id,
                    "split": str(
                        case[
                            "split"
                        ]
                    ),
                    "case_type": str(
                        case[
                            "case_type"
                        ]
                    ),
                    "query": str(
                        case[
                            "query"
                        ]
                    ),
                    "product_ids": [
                        int(value)
                        for value
                        in case[
                            "product_ids"
                        ]
                    ],
                    "error_type": type(
                        error
                    ).__name__,
                    "error": str(
                        error
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


def math_is_nan(
    value,
):
    try:
        return bool(
            np.isnan(
                value
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def comparison_failure_summary(
    frame,
):
    counter = Counter()

    if "failure_tags" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "failure_tag",
                "count",
            ]
        )

    for values in frame[
        "failure_tags"
    ]:
        if not isinstance(
            values,
            list,
        ):
            continue

        counter.update(
            values
        )

    return pd.DataFrame(
        [
            {
                "failure_tag": tag,
                "count": count,
            }
            for tag, count
            in counter.most_common()
        ]
    )


def build_comparison_report(
    frame,
):
    ok = frame[
        frame[
            "status"
        ]
        == "ok"
    ].copy()

    summary = summarize_comparison_results(
        ok
    )

    failures = comparison_failure_summary(
        ok
    )

    error_count = int(
        (
            frame[
                "status"
            ]
            != "ok"
        ).sum()
    )

    return {
        "case_count": int(
            len(
                frame
            )
        ),
        "successful_cases": int(
            len(
                ok
            )
        ),
        "error_count": error_count,
        "overall": summary[
            "overall"
        ],
        "telemetry": summary[
            "telemetry"
        ],
        "failure_counts": {
            str(
                row.failure_tag
            ): int(
                row.count
            )
            for row in failures.itertuples(
                index=False
            )
        },
    }


def export_manual_review_csv(
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
            "query",
            "product_ids",
            "summary",
            "overall_winner_product_id",
            "overall_recommendation",
            "confidence",
            "insufficient_evidence",
            "overall_score",
            "correctness",
            "groundedness",
            "criterion_coverage",
            "conflict_handling",
            "recommendation_calibration",
            "failure_tags",
            "judge_summary_reason",
        ]
        if column in frame.columns
    ]

    output = frame[
        columns
    ].copy()

    output[
        "human_score"
    ] = pd.NA
    output[
        "human_notes"
    ] = ""

    output.to_csv(
        path,
        index=False,
    )
