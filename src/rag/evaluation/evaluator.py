import time

import numpy as np
import pandas as pd

from .metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank_at_k,
    average_precision_at_k,
    ndcg_at_k
)


class RetrievalEvaluator:

    def __init__(
        self,
        retriever,
        dataset,
        name="retriever"
    ):
        self.retriever = retriever
        self.dataset = dataset
        self.name = name


    def evaluate(
        self,
        ks=(1, 3, 5, 10)
    ):
        ks = tuple(sorted(set(ks)))
        max_k = max(ks)

        rows = []

        for sample_idx, sample in enumerate(
            self.dataset
        ):
            start = time.perf_counter()

            results = self.retriever.retrieve(
                sample["query"],
                top_k=max_k,
                candidate_ids=sample[
                    "candidate_ids"
                ]
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            retrieved_ids = (
                results["id"]
                .dropna()
                .astype(int)
                .tolist()
            )

            relevant_ids = [
                int(comment_id)
                for comment_id in sample[
                    "relevant_ids"
                ]
            ]

            row = {
                "retriever": self.name,
                "sample_idx": sample_idx,
                "product_id": sample[
                    "product_id"
                ],
                "query": sample["query"],
                "candidate_count": len(
                    sample["candidate_ids"]
                ),
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "latency_ms": latency_ms
            }

            for k in ks:
                row[f"precision@{k}"] = (
                    precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                row[f"recall@{k}"] = (
                    recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                row[f"hit_rate@{k}"] = (
                    hit_rate_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                row[f"mrr@{k}"] = (
                    reciprocal_rank_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                row[f"map@{k}"] = (
                    average_precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

                row[f"ndcg@{k}"] = (
                    ndcg_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k
                    )
                )

            rows.append(row)

        per_query = pd.DataFrame(rows)
        summary = self.summarize(per_query)

        return summary, per_query


    @staticmethod
    def summarize(per_query):
        metric_columns = [
            column
            for column in per_query.columns
            if "@" in column
        ]

        summary = {
            metric: per_query[metric].mean()
            for metric in metric_columns
        }

        summary.update(
            {
                "latency_mean_ms": per_query[
                    "latency_ms"
                ].mean(),
                "latency_p50_ms": per_query[
                    "latency_ms"
                ].quantile(0.50),
                "latency_p95_ms": per_query[
                    "latency_ms"
                ].quantile(0.95)
            }
        )

        return summary


    @staticmethod
    def failure_cases(
        per_query,
        metric="recall@5",
        n=10
    ):
        return (
            per_query
            .sort_values(
                [metric, "latency_ms"],
                ascending=[True, False]
            )
            .head(n)
        )



def compare_retrievers(
    retrievers,
    dataset,
    ks=(1, 3, 5, 10)
):
    summaries = []
    per_query_frames = []

    for name, retriever in retrievers.items():
        evaluator = RetrievalEvaluator(
            retriever=retriever,
            dataset=dataset,
            name=name
        )

        summary, per_query = evaluator.evaluate(
            ks=ks
        )

        summaries.append(
            {
                "retriever": name,
                **summary
            }
        )

        per_query_frames.append(per_query)

    summary_df = (
        pd.DataFrame(summaries)
        .set_index("retriever")
    )

    per_query_df = pd.concat(
        per_query_frames,
        ignore_index=True
    )

    return summary_df, per_query_df



def bootstrap_confidence_intervals(
    per_query,
    metrics=(
        "recall@5",
        "mrr@5",
        "ndcg@5"
    ),
    n_bootstrap=2000,
    confidence=0.95,
    random_state=42
):
    rng = np.random.default_rng(
        random_state
    )

    rows = []

    for retriever, group in per_query.groupby(
        "retriever"
    ):
        for metric in metrics:
            values = group[metric].to_numpy(
                dtype=float
            )

            means = np.empty(
                n_bootstrap,
                dtype=float
            )

            for i in range(n_bootstrap):
                sample = rng.choice(
                    values,
                    size=len(values),
                    replace=True
                )

                means[i] = sample.mean()

            alpha = 1.0 - confidence

            rows.append(
                {
                    "retriever": retriever,
                    "metric": metric,
                    "mean": values.mean(),
                    "ci_low": np.quantile(
                        means,
                        alpha / 2
                    ),
                    "ci_high": np.quantile(
                        means,
                        1 - alpha / 2
                    )
                }
            )

    return pd.DataFrame(rows)
