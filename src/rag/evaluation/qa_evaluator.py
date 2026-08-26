import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .qa_metrics import (
    citation_validity,
    evidence_precision,
    evidence_recall,
    evidence_f1,
    retrieval_evidence_recall,
)


DEFAULT_WEIGHTS = {
    "correctness": 0.25,
    "relevance": 0.15,
    "completeness": 0.15,
    "groundedness": 0.25,
    "instruction_following": 0.15,
    "safety": 0.05,
}


class QAEvaluator:

    def __init__(
        self,
        qa_pipeline,
        judge,
        dataset,
        weights=None,
    ):
        self.qa_pipeline = (
            qa_pipeline
        )
        self.judge = judge
        self.dataset = dataset

        self.weights = dict(
            weights
            or DEFAULT_WEIGHTS
        )

        self._validate_weights()


    def _validate_weights(
        self,
    ):
        total = sum(
            self.weights.values()
        )

        if not np.isclose(
            total,
            1.0,
        ):
            raise ValueError(
                "Judge weights must sum to 1.0"
            )


    def _select_indices(
        self,
        max_samples=None,
        one_query_per_product=True,
        random_state=42,
    ):
        total = len(
            self.dataset.samples
        )

        indices = np.arange(
            total
        )

        rng = np.random.default_rng(
            random_state
        )

        if one_query_per_product:
            product_to_indices = {}

            for idx, sample in enumerate(
                self.dataset.samples
            ):
                product_id = int(
                    sample["product_id"]
                )

                product_to_indices.setdefault(
                    product_id,
                    [],
                ).append(idx)

            selected = []

            product_ids = list(
                product_to_indices
            )

            rng.shuffle(
                product_ids
            )

            for product_id in product_ids:
                choices = (
                    product_to_indices[
                        product_id
                    ]
                )

                selected.append(
                    int(
                        rng.choice(
                            choices
                        )
                    )
                )

            indices = np.asarray(
                selected,
                dtype=int,
            )

        else:
            indices = rng.permutation(
                indices
            )

        if max_samples is not None:
            indices = indices[
                :int(max_samples)
            ]

        return [
            int(x)
            for x in indices
        ]


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

        with open(
            path,
            encoding="utf-8",
        ) as file:
            for line in file:
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
                        int(
                            row["sample_idx"]
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

        with open(
            path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


    def _overall_score(
        self,
        judge_payload,
    ):
        return float(
            sum(
                self.weights[
                    dimension
                ]
                * float(
                    judge_payload[
                        dimension
                    ]["score"]
                )
                for dimension
                in self.weights
            )
        )


    def _evaluate_one(
        self,
        sample_idx,
    ):
        sample = (
            self.dataset.samples[
                sample_idx
            ]
        )

        # IMPORTANT:
        # use annotated candidate_ids for benchmark QA.
        # This keeps deterministic evidence precision/recall valid.
        qa_result = (
            self.qa_pipeline
            .answer(
                query=sample[
                    "query"
                ],
                product_id=sample[
                    "product_id"
                ],
                candidate_ids=sample[
                    "candidate_ids"
                ],
            )
        )

        judge_result = (
            self.judge.judge(
                query=sample[
                    "query"
                ],
                retrieved_documents=(
                    qa_result[
                        "retrieved_documents"
                    ]
                ),
                answer=qa_result[
                    "answer"
                ],
                evidence_ids=(
                    qa_result[
                        "evidence_ids"
                    ]
                ),
            )
        )

        judge_payload = (
            judge_result[
                "payload"
            ]
        )

        relevant_ids = [
            int(x)
            for x in sample[
                "relevant_ids"
            ]
        ]

        retrieved_ids = [
            int(x)
            for x in qa_result[
                "retrieved_ids"
            ]
        ]

        evidence_ids = [
            int(x)
            for x in qa_result[
                "evidence_ids"
            ]
        ]

        row = {
            "status": "ok",
            "sample_idx": int(
                sample_idx
            ),
            "product_id": int(
                sample[
                    "product_id"
                ]
            ),
            "product_title": (
                sample.get(
                    "product_title"
                )
            ),
            "query": sample[
                "query"
            ],
            "answer": qa_result[
                "answer"
            ],
            "confidence": qa_result[
                "confidence"
            ],
            "insufficient_evidence": (
                qa_result[
                    "insufficient_evidence"
                ]
            ),
            "relevant_ids": (
                relevant_ids
            ),
            "retrieved_ids": (
                retrieved_ids
            ),
            "evidence_ids": (
                evidence_ids
            ),

            "citation_validity": (
                citation_validity(
                    evidence_ids,
                    retrieved_ids,
                )
            ),
            "evidence_precision": (
                evidence_precision(
                    evidence_ids,
                    relevant_ids,
                )
            ),
            "evidence_recall": (
                evidence_recall(
                    evidence_ids,
                    relevant_ids,
                )
            ),
            "evidence_f1": (
                evidence_f1(
                    evidence_ids,
                    relevant_ids,
                )
            ),
            "retrieval_evidence_recall": (
                retrieval_evidence_recall(
                    retrieved_ids,
                    relevant_ids,
                )
            ),

            "overall_score": (
                self._overall_score(
                    judge_payload
                )
            ),

            "failure_tags": (
                judge_payload[
                    "failure_tags"
                ]
            ),
            "judge_summary_reason": (
                judge_payload[
                    "summary_reason"
                ]
            ),

            "qa_retrieval_latency_ms": (
                qa_result[
                    "telemetry"
                ][
                    "retrieval_latency_ms"
                ]
            ),
            "qa_generation_latency_ms": (
                qa_result[
                    "telemetry"
                ][
                    "generation_latency_ms"
                ]
            ),
            "qa_total_latency_ms": (
                qa_result[
                    "telemetry"
                ][
                    "total_latency_ms"
                ]
            ),
            "qa_prompt_tokens": (
                qa_result[
                    "telemetry"
                ][
                    "prompt_tokens"
                ]
            ),
            "qa_completion_tokens": (
                qa_result[
                    "telemetry"
                ][
                    "completion_tokens"
                ]
            ),
            "qa_total_tokens": (
                qa_result[
                    "telemetry"
                ][
                    "total_tokens"
                ]
            ),
            "qa_cost_usd": (
                qa_result[
                    "telemetry"
                ][
                    "estimated_cost_usd"
                ]
            ),

            "judge_latency_ms": (
                judge_result[
                    "telemetry"
                ][
                    "latency_ms"
                ]
            ),
            "judge_prompt_tokens": (
                judge_result[
                    "telemetry"
                ][
                    "prompt_tokens"
                ]
            ),
            "judge_completion_tokens": (
                judge_result[
                    "telemetry"
                ][
                    "completion_tokens"
                ]
            ),
            "judge_total_tokens": (
                judge_result[
                    "telemetry"
                ][
                    "total_tokens"
                ]
            ),
            "judge_cost_usd": (
                judge_result[
                    "telemetry"
                ][
                    "estimated_cost_usd"
                ]
            ),
        }

        for dimension in self.weights:
            row[
                dimension
            ] = float(
                judge_payload[
                    dimension
                ]["score"]
            )

            row[
                f"{dimension}_reason"
            ] = (
                judge_payload[
                    dimension
                ]["reason"]
            )

        return row


    def evaluate(
        self,
        output_path,
        max_samples=30,
        one_query_per_product=True,
        random_state=42,
        resume=True,
        continue_on_error=True,
        refresh_invalid_citations=True,
    ):
        output_path = Path(
            output_path
        )

        selected_indices = (
            self._select_indices(
                max_samples=(
                    max_samples
                ),
                one_query_per_product=(
                    one_query_per_product
                ),
                random_state=(
                    random_state
                ),
            )
        )

        completed = {}

        if resume:
            completed = (
                self._load_checkpoint(
                    output_path
                )
            )

        rows = []

        for progress, sample_idx in enumerate(
            selected_indices,
            start=1,
        ):
            if sample_idx in completed:
                row = completed[
                    sample_idx
                ]

                citation_is_valid = (
                    float(
                        row.get(
                            "citation_validity",
                            0.0,
                        )
                    )
                    >= 1.0
                )

                if (
                    not refresh_invalid_citations
                    or citation_is_valid
                ):
                    rows.append(
                        row
                    )

                    print(
                        f"[{progress}/"
                        f"{len(selected_indices)}] "
                        f"sample={sample_idx} "
                        "cached"
                    )

                    continue

                print(
                    f"[{progress}/"
                    f"{len(selected_indices)}] "
                    f"sample={sample_idx} "
                    "refreshing invalid citation"
                )

            try:
                row = (
                    self._evaluate_one(
                        sample_idx
                    )
                )

                rows.append(
                    row
                )

                self._append_checkpoint(
                    output_path,
                    row,
                )

                print(
                    f"[{progress}/"
                    f"{len(selected_indices)}] "
                    f"sample={sample_idx} "
                    f"score="
                    f"{row['overall_score']:.2f}"
                )

            except Exception as exc:
                error_row = {
                    "status": "error",
                    "sample_idx": int(
                        sample_idx
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

                self._append_checkpoint(
                    output_path,
                    error_row,
                )

                print(
                    f"[{progress}/"
                    f"{len(selected_indices)}] "
                    f"sample={sample_idx} "
                    f"ERROR: {exc}"
                )

                if not continue_on_error:
                    raise

        successful = [
            row
            for row in rows
            if row.get(
                "status"
            ) == "ok"
        ]

        results = pd.DataFrame(
            successful
        )

        summary = self.summarize(
            results
        )

        return (
            summary,
            results,
        )


    @staticmethod
    def _mean_nullable(
        series,
    ):
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        if values.notna().sum() == 0:
            return None

        return float(
            values.mean()
        )


    @staticmethod
    def _sum_nullable(
        series,
    ):
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        if values.notna().sum() == 0:
            return None

        return float(
            values.sum()
        )


    def summarize(
        self,
        results,
    ):
        if len(results) == 0:
            return {
                "samples": 0
            }

        summary = {
            "samples": int(
                len(results)
            ),
            "overall_score": float(
                results[
                    "overall_score"
                ].mean()
            ),
        }

        for dimension in self.weights:
            summary[
                dimension
            ] = float(
                results[
                    dimension
                ].mean()
            )

        summary.update(
            {
                "citation_validity": float(
                    results[
                        "citation_validity"
                    ].mean()
                ),
                "evidence_precision": float(
                    results[
                        "evidence_precision"
                    ].mean()
                ),
                "evidence_recall": float(
                    results[
                        "evidence_recall"
                    ].mean()
                ),
                "evidence_f1": float(
                    results[
                        "evidence_f1"
                    ].mean()
                ),
                "retrieval_evidence_recall": float(
                    results[
                        "retrieval_evidence_recall"
                    ].mean()
                ),

                # Production QA metrics only.
                "latency_avg_sec": float(
                    results[
                        "qa_total_latency_ms"
                    ].mean()
                    / 1000
                ),
                "latency_p50_sec": float(
                    results[
                        "qa_total_latency_ms"
                    ].quantile(
                        0.50
                    )
                    / 1000
                ),
                "latency_p95_sec": float(
                    results[
                        "qa_total_latency_ms"
                    ].quantile(
                        0.95
                    )
                    / 1000
                ),
                "avg_tokens": float(
                    results[
                        "qa_total_tokens"
                    ].mean()
                ),
                "avg_cost_usd": (
                    self._mean_nullable(
                        results[
                            "qa_cost_usd"
                        ]
                    )
                ),
                "total_cost_usd": (
                    self._sum_nullable(
                        results[
                            "qa_cost_usd"
                        ]
                    )
                ),

                # Judge is evaluation overhead, not product latency.
                "judge_latency_avg_sec": float(
                    results[
                        "judge_latency_ms"
                    ].mean()
                    / 1000
                ),
                "judge_avg_tokens": float(
                    results[
                        "judge_total_tokens"
                    ].mean()
                ),
                "judge_avg_cost_usd": (
                    self._mean_nullable(
                        results[
                            "judge_cost_usd"
                        ]
                    )
                ),
                "judge_total_cost_usd": (
                    self._sum_nullable(
                        results[
                            "judge_cost_usd"
                        ]
                    )
                ),
            }
        )

        all_tags = []

        for tags in results[
            "failure_tags"
        ]:
            if isinstance(
                tags,
                list,
            ):
                all_tags.extend(
                    tags
                )

        summary[
            "failure_tag_counts"
        ] = dict(
            Counter(
                all_tags
            )
        )

        return summary


    @staticmethod
    def failure_cases(
        results,
        n=10,
    ):
        columns = [
            "sample_idx",
            "product_title",
            "query",
            "answer",
            "overall_score",
            "correctness",
            "groundedness",
            "evidence_precision",
            "evidence_recall",
            "failure_tags",
            "judge_summary_reason",
        ]

        available = [
            column
            for column in columns
            if column
            in results.columns
        ]

        return (
            results
            .sort_values(
                [
                    "overall_score",
                    "groundedness",
                    "correctness",
                ],
                ascending=True,
            )
            .head(n)[available]
        )


def format_qa_summary(
    summary,
):
    def score(
        key,
    ):
        return (
            f"{summary[key]:.2f}"
        )

    def percent(
        key,
    ):
        return (
            f"{100 * summary[key]:.1f}%"
        )

    avg_cost = (
        "N/A"
        if summary[
            "avg_cost_usd"
        ] is None
        else (
            f"${summary['avg_cost_usd']:.4f}"
        )
    )

    total_cost = (
        "N/A"
        if summary[
            "total_cost_usd"
        ] is None
        else (
            f"${summary['total_cost_usd']:.4f}"
        )
    )

    judge_cost = (
        "N/A"
        if summary[
            "judge_total_cost_usd"
        ] is None
        else (
            f"${summary['judge_total_cost_usd']:.4f}"
        )
    )

    lines = [
        "GROUNDING QA EVALUATION",
        "=======================",
        "",
        f"Samples:               {summary['samples']}",
        "",
        (
            "Overall Score:         "
            f"{score('overall_score')} / 5"
        ),
        "",
        (
            "Correctness:           "
            f"{score('correctness')}"
        ),
        (
            "Relevance:             "
            f"{score('relevance')}"
        ),
        (
            "Completeness:          "
            f"{score('completeness')}"
        ),
        (
            "Groundedness:          "
            f"{score('groundedness')}"
        ),
        (
            "Instruction Following: "
            f"{score('instruction_following')}"
        ),
        (
            "Safety:                "
            f"{score('safety')}"
        ),
        "",
        (
            "Citation Validity:     "
            f"{percent('citation_validity')}"
        ),
        (
            "Evidence Precision:    "
            f"{percent('evidence_precision')}"
        ),
        (
            "Evidence Recall:       "
            f"{percent('evidence_recall')}"
        ),
        (
            "Evidence F1:           "
            f"{percent('evidence_f1')}"
        ),
        (
            "Retrieval Evidence R:  "
            f"{percent('retrieval_evidence_recall')}"
        ),
        "",
        (
            "Avg Latency:           "
            f"{summary['latency_avg_sec']:.2f} sec"
        ),
        (
            "P50 Latency:           "
            f"{summary['latency_p50_sec']:.2f} sec"
        ),
        (
            "P95 Latency:           "
            f"{summary['latency_p95_sec']:.2f} sec"
        ),
        (
            "Avg Tokens:            "
            f"{summary['avg_tokens']:.0f}"
        ),
        (
            "Avg Cost:              "
            f"{avg_cost}"
        ),
        (
            "Total QA Cost:         "
            f"{total_cost}"
        ),
        "",
        "LLM-AS-A-JUDGE OVERHEAD",
        "-----------------------",
        (
            "Avg Judge Latency:     "
            f"{summary['judge_latency_avg_sec']:.2f} sec"
        ),
        (
            "Avg Judge Tokens:      "
            f"{summary['judge_avg_tokens']:.0f}"
        ),
        (
            "Total Judge Cost:      "
            f"{judge_cost}"
        ),
    ]

    return "\n".join(
        lines
    )
