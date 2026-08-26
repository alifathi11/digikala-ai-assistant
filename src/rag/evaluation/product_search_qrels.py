from pathlib import Path
import re
import time

import pandas as pd


class ProductSearchQrelsBuilder:
    RESCUE_STOPWORDS = {
        "و", "یا", "با", "برای", "که", "رو", "را", "به", "از", "در",
        "مناسب", "خوب", "سبک", "راحت", "طولانی", "حداقل", "ظرفیت",
        "مدل", "حجم", "نکنه", "نشه", "بدون",
    }

    def __init__(
        self,
        metadata_retriever,
        review_evidence,
        judge,
        hybrid_pool_k=12,
        dense_pool_k=4,
        sparse_pool_k=4,
        rescue_pool_k=8,
        rescue_fragment_k=3,
        rescue_max_fragments=8,
    ):
        self.metadata = metadata_retriever
        self.review_evidence = review_evidence
        self.judge = judge
        self.hybrid_pool_k = int(hybrid_pool_k)
        self.dense_pool_k = int(dense_pool_k)
        self.sparse_pool_k = int(sparse_pool_k)
        self.rescue_pool_k = int(rescue_pool_k)
        self.rescue_fragment_k = int(rescue_fragment_k)
        self.rescue_max_fragments = int(rescue_max_fragments)

    def _rescue_fragments(self, query):
        processed = self.metadata._process(query)

        tokens = [
            token
            for token in re.findall(
                r"[^\W_]+",
                str(processed),
                flags=re.UNICODE,
            )
            if (
                token
                and token not in self.RESCUE_STOPWORDS
                and not token.isdigit()
            )
        ]

        fragments = []

        for width in (3, 2, 1):
            if len(tokens) < width:
                continue

            for start in range(
                len(tokens) - width + 1
            ):
                fragment = " ".join(
                    tokens[
                        start:
                        start + width
                    ]
                ).strip()

                if (
                    fragment
                    and fragment not in fragments
                ):
                    fragments.append(fragment)

                if (
                    len(fragments)
                    >= self.rescue_max_fragments
                ):
                    return fragments

        return fragments

    def _rescue_pool(self, query):
        rows = []

        for fragment_index, fragment in enumerate(
            self._rescue_fragments(query),
            start=1,
        ):
            retrieved = (
                self.metadata
                .sparse_index
                .retrieve(
                    fragment,
                    top_k=self.rescue_fragment_k,
                )
                .copy()
                .reset_index(drop=True)
            )

            fragment_width = len(
                fragment.split()
            )

            for rank, row in enumerate(
                retrieved.itertuples(index=False),
                start=1,
            ):
                values = row._asdict()

                rows.append(
                    {
                        **values,
                        "rescue_fragment": fragment,
                        "rescue_fragment_width": fragment_width,
                        "rescue_fragment_rank": rank,
                        "rescue_priority": (
                            10000 * fragment_width
                            - 100 * fragment_index
                            - rank
                        ),
                    }
                )

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)

        frame = (
            frame
            .sort_values(
                [
                    "rescue_priority",
                    "score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .drop_duplicates(
                subset=["id"],
                keep="first",
            )
            .head(self.rescue_pool_k)
            .reset_index(drop=True)
        )

        frame["rescue_pool_rank"] = range(
            1,
            len(frame) + 1,
        )

        return frame

    def _pool(self, query):
        hybrid = self.metadata.retrieve(query, top_k=self.hybrid_pool_k).reset_index(drop=True)
        processed = self.metadata._process(query)
        query_embedding = self.metadata.embedding_model.encode([processed])
        dense = self.metadata.dense_index.search(query_embedding, top_k=self.dense_pool_k).reset_index(drop=True)
        sparse = self.metadata.sparse_index.retrieve(query, top_k=self.sparse_pool_k).reset_index(drop=True)

        ranks = {}
        for name, frame in [("hybrid", hybrid), ("dense", dense), ("sparse", sparse)]:
            ranks[name] = {int(product_id): rank for rank, product_id in enumerate(frame["id"], start=1)}

        pool = pd.concat([hybrid, dense, sparse], ignore_index=True).drop_duplicates("id").reset_index(drop=True)
        for name in ranks:
            pool[f"{name}_pool_rank"] = pool["id"].astype(int).map(ranks[name])

        pool["_priority"] = pool[["hybrid_pool_rank", "dense_pool_rank", "sparse_pool_rank"]].min(axis=1, skipna=True)
        return pool.sort_values("_priority").drop(columns="_priority").reset_index(drop=True)

    def build(self, queries, qrels_path, telemetry_path, manual_review_path, resume=True):
        qrels_path = Path(qrels_path)
        telemetry_path = Path(telemetry_path)
        manual_review_path = Path(manual_review_path)
        qrels_path.parent.mkdir(parents=True, exist_ok=True)

        qrels = pd.read_parquet(qrels_path) if resume and qrels_path.exists() else pd.DataFrame()
        telemetry = pd.read_csv(telemetry_path) if resume and telemetry_path.exists() else pd.DataFrame()
        completed = set(qrels["query_id"].astype(str).unique()) if len(qrels) else set()
        qrel_parts = [qrels] if len(qrels) else []
        telemetry_parts = [telemetry] if len(telemetry) else []

        for row in queries.itertuples(index=False):
            query_id = str(row.query_id)
            if resume and query_id in completed:
                continue

            start = time.perf_counter()
            pool = self._pool(row.query)
            reviews, review_tel = self.review_evidence.retrieve(
                query=row.query,
                product_ids=pool["id"].astype(int).tolist(),
            )
            judgments, judge_tel = self.judge.judge(row.query, pool, reviews)
            judged = pool.merge(judgments, left_on="id", right_on="product_id", how="left")
            judged["query_id"] = query_id
            judged["query_type"] = str(row.query_type)
            judged["query"] = str(row.query)
            judged["split"] = str(row.split)
            judged["gold_grade"] = pd.NA
            judged["gold_notes"] = ""
            judged["manual_review_priority"] = (
                (judged["split"] == "test")
                | (judged["teacher_confidence"] != "high")
                | judged["teacher_grade"].isin([1, 2])
                | (judged["hybrid_pool_rank"].fillna(999) <= 5)
            )
            qrel_parts.append(judged)

            telemetry_parts.append(pd.DataFrame([{
                "query_id": query_id,
                "query_type": str(row.query_type),
                "split": str(row.split),
                "pool_size": len(pool),
                "total_latency_ms": (time.perf_counter() - start) * 1000,
                **{f"review_{k}": v for k, v in review_tel.items()},
                **{f"judge_{k}": v for k, v in judge_tel.items()},
            }]))

            pd.concat(qrel_parts, ignore_index=True).to_parquet(qrels_path, index=False)
            pd.concat(telemetry_parts, ignore_index=True).to_csv(telemetry_path, index=False)

        qrels = pd.concat(qrel_parts, ignore_index=True)
        telemetry = pd.concat(telemetry_parts, ignore_index=True)

        columns = [
            "query_id", "query_type", "query", "split", "id", "title_fa", "Brand", "Category2",
            "hybrid_pool_rank", "dense_pool_rank", "sparse_pool_rank",
            "teacher_grade", "teacher_confidence", "teacher_reason", "teacher_evidence_ids",
            "manual_review_priority", "gold_grade", "gold_notes",
        ]
        columns = [column for column in columns if column in qrels.columns]
        qrels[columns].sort_values(
            ["manual_review_priority", "query_id", "hybrid_pool_rank"],
            ascending=[False, True, True], na_position="last"
        ).to_csv(manual_review_path, index=False)
        return {"qrels": qrels, "telemetry": telemetry}


    def augment_with_rescue(
        self,
        queries,
        qrels_path,
        telemetry_path,
        manual_review_path,
    ):
        """
        Add only NEW evaluation-rescue candidates.

        Existing qrels are preserved, so previous judge calls are not repeated.
        Production retrieval is not changed.
        """
        qrels_path = Path(qrels_path)
        telemetry_path = Path(
            telemetry_path
        )
        manual_review_path = Path(
            manual_review_path
        )

        if not qrels_path.exists():
            raise FileNotFoundError(
                "qrels.parquet not found."
            )

        qrels = pd.read_parquet(
            qrels_path
        )

        telemetry = (
            pd.read_csv(
                telemetry_path
            )
            if telemetry_path.exists()
            else pd.DataFrame()
        )

        new_parts = []
        telemetry_parts = (
            [telemetry]
            if len(telemetry)
            else []
        )

        new_rows = 0

        for row in queries.itertuples(
            index=False
        ):
            query_id = str(
                row.query_id
            )

            existing_ids = set(
                qrels[
                    qrels[
                        "query_id"
                    ].astype(str)
                    == query_id
                ][
                    "id"
                ]
                .astype(int)
                .tolist()
            )

            rescue = self._rescue_pool(
                row.query
            )

            if len(rescue) == 0:
                continue

            rescue = rescue[
                ~rescue[
                    "id"
                ]
                .astype(int)
                .isin(existing_ids)
            ].copy()

            if len(rescue) == 0:
                continue

            rescue[
                "hybrid_pool_rank"
            ] = pd.NA
            rescue[
                "dense_pool_rank"
            ] = pd.NA
            rescue[
                "sparse_pool_rank"
            ] = pd.NA

            start = time.perf_counter()

            reviews, review_tel = (
                self.review_evidence
                .retrieve(
                    query=row.query,
                    product_ids=(
                        rescue[
                            "id"
                        ]
                        .astype(int)
                        .tolist()
                    ),
                )
            )

            judgments, judge_tel = (
                self.judge.judge(
                    row.query,
                    rescue,
                    reviews,
                )
            )

            judged = rescue.merge(
                judgments,
                left_on="id",
                right_on="product_id",
                how="left",
            )

            judged[
                "query_id"
            ] = query_id
            judged[
                "query_type"
            ] = str(
                row.query_type
            )
            judged[
                "query"
            ] = str(
                row.query
            )
            judged[
                "split"
            ] = str(
                row.split
            )
            judged[
                "gold_grade"
            ] = pd.NA
            judged[
                "gold_notes"
            ] = ""
            judged[
                "manual_review_priority"
            ] = True

            new_parts.append(judged)
            new_rows += len(judged)

            telemetry_parts.append(
                pd.DataFrame(
                    [
                        {
                            "query_id": query_id,
                            "query_type": str(
                                row.query_type
                            ),
                            "split": str(
                                row.split
                            ),
                            "phase": (
                                "rescue_augmentation"
                            ),
                            "pool_size": len(
                                rescue
                            ),
                            "total_latency_ms": (
                                (
                                    time.perf_counter()
                                    - start
                                )
                                * 1000
                            ),
                            **{
                                f"review_{key}": value
                                for key, value
                                in review_tel.items()
                            },
                            **{
                                f"judge_{key}": value
                                for key, value
                                in judge_tel.items()
                            },
                        }
                    ]
                )
            )

        if new_parts:
            qrels = pd.concat(
                [
                    qrels,
                    *new_parts,
                ],
                ignore_index=True,
            )

            qrels = (
                qrels
                .drop_duplicates(
                    subset=[
                        "query_id",
                        "id",
                    ],
                    keep="first",
                )
                .reset_index(drop=True)
            )

        telemetry = (
            pd.concat(
                telemetry_parts,
                ignore_index=True,
            )
            if telemetry_parts
            else pd.DataFrame()
        )

        qrels.to_parquet(
            qrels_path,
            index=False,
        )

        telemetry.to_csv(
            telemetry_path,
            index=False,
        )

        review_columns = [
            "query_id",
            "query_type",
            "query",
            "split",
            "id",
            "title_fa",
            "Brand",
            "Category2",
            "hybrid_pool_rank",
            "dense_pool_rank",
            "sparse_pool_rank",
            "rescue_pool_rank",
            "rescue_fragment",
            "teacher_grade",
            "teacher_confidence",
            "teacher_reason",
            "teacher_evidence_ids",
            "manual_review_priority",
            "gold_grade",
            "gold_notes",
        ]

        review_columns = [
            column
            for column in review_columns
            if column in qrels.columns
        ]

        qrels[
            review_columns
        ].sort_values(
            [
                "manual_review_priority",
                "query_id",
                "hybrid_pool_rank",
                "rescue_pool_rank",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        ).to_csv(
            manual_review_path,
            index=False,
        )

        return {
            "qrels": qrels,
            "telemetry": telemetry,
            "new_qrel_rows": int(
                new_rows
            ),
        }


def resolve_qrels(qrels, allow_teacher_proxy=True):
    frame = qrels.copy()
    gold = pd.to_numeric(frame["gold_grade"], errors="coerce")
    teacher = pd.to_numeric(frame["teacher_grade"], errors="coerce")

    if allow_teacher_proxy:
        frame["relevance_grade"] = gold.fillna(teacher)
        frame["label_source"] = "llm_teacher_proxy"
        frame.loc[gold.notna(), "label_source"] = "manual_gold"
    else:
        if gold.isna().any():
            raise ValueError("Manual gold is incomplete; set allow_teacher_proxy=true or complete gold_grade.")
        frame["relevance_grade"] = gold
        frame["label_source"] = "manual_gold"

    frame["relevance_grade"] = frame["relevance_grade"].clip(0, 3).astype(int)
    return frame
