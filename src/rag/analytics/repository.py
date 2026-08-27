
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd



def _canonicalize_products_for_analytics(products):
    """Collapse seller-level rows to one row per product ID.

    Kept local to analytics so this deterministic layer does not import the
    FAISS/Tantivy product-search package just to prepare tabular data.
    """
    frame = products.copy().reset_index(drop=True)

    if "id" not in frame.columns:
        raise ValueError("products requires an id column")

    frame["id"] = pd.to_numeric(frame["id"], errors="coerce")
    frame = frame[frame["id"].notna()].copy()
    frame["id"] = frame["id"].astype("int64")

    rate_cnt = pd.to_numeric(
        frame.get("Rate_cnt", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)

    price = pd.to_numeric(
        frame.get("Price", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )

    frame["_analytics_sort_rate_cnt"] = rate_cnt
    frame["_analytics_sort_price"] = price.fillna(float("inf"))

    representative = (
        frame
        .sort_values(
            ["id", "_analytics_sort_rate_cnt", "_analytics_sort_price"],
            ascending=[True, False, True],
        )
        .drop_duplicates("id", keep="first")
        .drop(columns=["_analytics_sort_rate_cnt", "_analytics_sort_price"])
        .set_index("id")
    )

    grouped = frame.groupby("id", sort=False)
    representative["source_row_count"] = grouped.size()

    if "Seller" in frame.columns:
        representative["seller_count"] = grouped["Seller"].nunique(dropna=True)
    else:
        representative["seller_count"] = grouped.size()

    if "Price" in frame.columns:
        representative["Price"] = (
            pd.to_numeric(frame["Price"], errors="coerce")
            .groupby(frame["id"])
            .min()
        )

    if "min_price_last_month" in frame.columns:
        representative["min_price_last_month"] = (
            pd.to_numeric(frame["min_price_last_month"], errors="coerce")
            .groupby(frame["id"])
            .min()
        )

    return representative.reset_index()


class AnalyticsRepository:
    """
    Product-level analytics repository.

    Products MUST be canonical: one row per product ID.
    Comment data is scanned in Parquet batches and reduced to per-product
    aggregates so manager analytics never counts seller-level product rows
    more than once and never needs to load all comments into memory.
    """

    def __init__(
        self,
        products,
        comments_path=None,
        review_stats=None,
        product_source="in_memory",
    ):
        frame = (
            products
            .copy()
            .reset_index(drop=True)
        )

        if "id" not in frame.columns:
            raise ValueError(
                "Analytics products require an id column."
            )

        frame["id"] = pd.to_numeric(
            frame["id"],
            errors="coerce",
        )

        frame = frame[
            frame["id"].notna()
        ].copy()

        frame["id"] = (
            frame["id"]
            .astype("int64")
        )

        if frame[
            "id"
        ].duplicated().any():
            duplicate_count = int(
                frame[
                    "id"
                ].duplicated().sum()
            )

            raise ValueError(
                "Analytics requires canonical product rows; "
                f"found {duplicate_count} duplicate product IDs."
            )

        for column in (
            "Price",
            "min_price_last_month",
            "Rate",
            "Rate_cnt",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        self.products = frame
        self.comments_path = (
            Path(comments_path)
            if comments_path is not None
            else None
        )
        self.product_source = str(
            product_source
        )

        self._review_stats = None
        self._review_scan_summary = None

        if review_stats is not None:
            self._review_stats = (
                self._prepare_review_stats(
                    review_stats
                )
            )

            matched_reviews = int(
                self._review_stats[
                    "review_count"
                ].sum()
            )

            valid_rate_rows = int(
                self._review_stats[
                    "review_rate_count"
                ].sum()
            )

            self._review_scan_summary = {
                "source": "in_memory_review_stats",
                "total_comment_rows": (
                    matched_reviews
                ),
                "valid_product_id_rows": (
                    matched_reviews
                ),
                "matched_product_rows": (
                    matched_reviews
                ),
                "orphan_product_rows": 0,
                "product_join_rate": 1.0,
                "valid_review_rate_rows": (
                    valid_rate_rows
                ),
                "review_rate_coverage": (
                    valid_rate_rows
                    / matched_reviews
                    if matched_reviews
                    else 0.0
                ),
            }


    @classmethod
    def from_project_root(
        cls,
        project_root,
        products_path=None,
        comments_path=None,
    ):
        project_root = Path(
            project_root
        )

        if products_path is None:
            canonical_path = (
                project_root
                / "data"
                / "processed"
                / "products_search.parquet"
            )

            clean_path = (
                project_root
                / "data"
                / "processed"
                / "products_clean.parquet"
            )

            if canonical_path.exists():
                products_path = canonical_path
                product_source = (
                    "products_search.parquet"
                )
                products = pd.read_parquet(
                    products_path
                )
            elif clean_path.exists():
                products_path = clean_path
                product_source = (
                    "products_clean.parquet"
                    " -> canonicalized in memory"
                )

                products = (
                    _canonicalize_products_for_analytics(
                        pd.read_parquet(
                            products_path
                        )
                    )
                )
            else:
                raise FileNotFoundError(
                    "Could not find either "
                    "data/processed/products_search.parquet "
                    "or products_clean.parquet."
                )
        else:
            products_path = Path(
                products_path
            )

            if not products_path.exists():
                raise FileNotFoundError(
                    products_path
                )

            products = pd.read_parquet(
                products_path
            )

            product_source = str(
                products_path
            )

            if products[
                "id"
            ].duplicated().any():
                products = (
                    _canonicalize_products_for_analytics(
                        products
                    )
                )

                product_source += (
                    " -> canonicalized in memory"
                )

        if comments_path is None:
            comments_path = (
                project_root
                / "data"
                / "processed"
                / "comments_clean.parquet"
            )
        else:
            comments_path = Path(
                comments_path
            )

        if not comments_path.exists():
            raise FileNotFoundError(
                comments_path
            )

        return cls(
            products=products,
            comments_path=comments_path,
            product_source=(
                product_source
            ),
        )


    @staticmethod
    def _prepare_review_stats(
        review_stats,
    ):
        frame = (
            review_stats
            .copy()
            .reset_index(drop=True)
        )

        if "product_id" not in frame.columns:
            raise ValueError(
                "review_stats requires product_id."
            )

        frame[
            "product_id"
        ] = pd.to_numeric(
            frame[
                "product_id"
            ],
            errors="coerce",
        )

        frame = frame[
            frame[
                "product_id"
            ].notna()
        ].copy()

        frame[
            "product_id"
        ] = (
            frame[
                "product_id"
            ]
            .astype("int64")
        )

        defaults = {
            "review_count": 0,
            "review_rate_count": 0,
            "review_rate_sum": 0.0,
        }

        for column, default in (
            defaults.items()
        ):
            if column not in frame.columns:
                frame[column] = default

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).fillna(
                default
            )

        if (
            "avg_review_rate"
            not in frame.columns
        ):
            denominator = (
                frame[
                    "review_rate_count"
                ]
                .replace(
                    0,
                    pd.NA,
                )
            )

            frame[
                "avg_review_rate"
            ] = (
                frame[
                    "review_rate_sum"
                ]
                / denominator
            )

        return (
            frame[
                [
                    "product_id",
                    "review_count",
                    "review_rate_count",
                    "review_rate_sum",
                    "avg_review_rate",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "product_id"
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )


    def _scan_comments(
        self,
        batch_size=250_000,
    ):
        if self.comments_path is None:
            raise ValueError(
                "No comments_path is configured."
            )

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required to scan comments_clean.parquet. "
                "Install project requirements before running Notebook 16."
            ) from exc

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Scanning comments_clean.parquet requires pyarrow. "
                "Install project requirements before running Notebook 16."
            ) from exc

        parquet = pq.ParquetFile(
            self.comments_path
        )

        schema_names = set(
            parquet.schema.names
        )

        if (
            "product_id"
            not in schema_names
        ):
            raise ValueError(
                "comments parquet requires product_id."
            )

        columns = [
            "product_id"
        ]

        if "rate" in schema_names:
            columns.append(
                "rate"
            )

        product_ids = set(
            self.products[
                "id"
            ]
            .astype(int)
            .tolist()
        )

        review_counts = Counter()
        rate_counts = Counter()
        rate_sums = defaultdict(float)

        total_rows = 0
        valid_product_rows = 0
        matched_rows = 0
        orphan_rows = 0
        valid_rate_rows = 0

        for batch in parquet.iter_batches(
            batch_size=int(
                batch_size
            ),
            columns=columns,
        ):
            frame = batch.to_pandas()

            total_rows += int(
                len(frame)
            )

            ids = pd.to_numeric(
                frame[
                    "product_id"
                ],
                errors="coerce",
            )

            valid_id_mask = (
                ids.notna()
            )

            valid_product_rows += int(
                valid_id_mask.sum()
            )

            working = pd.DataFrame(
                {
                    "product_id": ids[
                        valid_id_mask
                    ].astype(
                        "int64"
                    )
                }
            )

            if len(working) == 0:
                continue

            match_mask = (
                working[
                    "product_id"
                ]
                .isin(
                    product_ids
                )
            )

            matched = working[
                match_mask
            ].copy()

            matched_rows += int(
                len(matched)
            )

            orphan_rows += int(
                (
                    ~match_mask
                ).sum()
            )

            batch_counts = (
                matched[
                    "product_id"
                ]
                .value_counts()
            )

            review_counts.update(
                {
                    int(
                        product_id
                    ): int(
                        count
                    )
                    for product_id, count
                    in batch_counts.items()
                }
            )

            if "rate" not in frame.columns:
                continue

            valid_source_index = (
                frame.index[
                    valid_id_mask
                ]
            )

            rates = pd.to_numeric(
                frame.loc[
                    valid_source_index,
                    "rate",
                ],
                errors="coerce",
            )

            rate_frame = pd.DataFrame(
                {
                    "product_id": (
                        working[
                            "product_id"
                        ].to_numpy()
                    ),
                    "rate": (
                        rates.to_numpy()
                    ),
                }
            )

            rate_frame = rate_frame[
                rate_frame[
                    "product_id"
                ].isin(
                    product_ids
                )
            ].copy()

            valid_rate_mask = (
                rate_frame[
                    "rate"
                ]
                .between(
                    1,
                    5,
                    inclusive="both",
                )
            )

            valid_rates = rate_frame[
                valid_rate_mask
            ]

            valid_rate_rows += int(
                len(valid_rates)
            )

            if len(valid_rates) == 0:
                continue

            grouped = (
                valid_rates
                .groupby(
                    "product_id"
                )[
                    "rate"
                ]
                .agg(
                    [
                        "count",
                        "sum",
                    ]
                )
            )

            rate_counts.update(
                {
                    int(
                        product_id
                    ): int(
                        row[
                            "count"
                        ]
                    )
                    for product_id, row
                    in grouped.iterrows()
                }
            )

            for product_id, row in (
                grouped.iterrows()
            ):
                rate_sums[
                    int(
                        product_id
                    )
                ] += float(
                    row[
                        "sum"
                    ]
                )

        all_ids = set(
            review_counts
        ) | set(
            rate_counts
        )

        rows = []

        for product_id in all_ids:
            review_count = int(
                review_counts.get(
                    product_id,
                    0,
                )
            )

            rate_count = int(
                rate_counts.get(
                    product_id,
                    0,
                )
            )

            rate_sum = float(
                rate_sums.get(
                    product_id,
                    0.0,
                )
            )

            rows.append(
                {
                    "product_id": (
                        int(
                            product_id
                        )
                    ),
                    "review_count": (
                        review_count
                    ),
                    "review_rate_count": (
                        rate_count
                    ),
                    "review_rate_sum": (
                        rate_sum
                    ),
                    "avg_review_rate": (
                        rate_sum
                        / rate_count
                        if rate_count
                        else pd.NA
                    ),
                }
            )

        stats = pd.DataFrame(
            rows
        )

        if len(stats) == 0:
            stats = pd.DataFrame(
                columns=[
                    "product_id",
                    "review_count",
                    "review_rate_count",
                    "review_rate_sum",
                    "avg_review_rate",
                ]
            )

        self._review_stats = (
            self._prepare_review_stats(
                stats
            )
        )

        self._review_scan_summary = {
            "source": str(
                self.comments_path
            ),
            "total_comment_rows": int(
                total_rows
            ),
            "valid_product_id_rows": int(
                valid_product_rows
            ),
            "matched_product_rows": int(
                matched_rows
            ),
            "orphan_product_rows": int(
                orphan_rows
            ),
            "product_join_rate": (
                matched_rows
                / valid_product_rows
                if valid_product_rows
                else 0.0
            ),
            "valid_review_rate_rows": int(
                valid_rate_rows
            ),
            "review_rate_coverage": (
                valid_rate_rows
                / matched_rows
                if matched_rows
                else 0.0
            ),
        }


    def review_stats(
        self,
        batch_size=250_000,
    ):
        if self._review_stats is None:
            self._scan_comments(
                batch_size=batch_size
            )

        return (
            self._review_stats
            .copy()
        )


    def review_scan_summary(
        self,
        batch_size=250_000,
    ):
        if (
            self._review_scan_summary
            is None
        ):
            self._scan_comments(
                batch_size=batch_size
            )

        return dict(
            self._review_scan_summary
        )


    def enriched_products(
        self,
        batch_size=250_000,
    ):
        stats = self.review_stats(
            batch_size=batch_size
        )

        frame = (
            self.products
            .merge(
                stats,
                left_on="id",
                right_on="product_id",
                how="left",
            )
            .drop(
                columns=[
                    "product_id"
                ],
                errors="ignore",
            )
        )

        for column in (
            "review_count",
            "review_rate_count",
            "review_rate_sum",
        ):
            if column not in frame.columns:
                frame[column] = 0

            frame[column] = (
                pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )
                .fillna(0)
            )

        frame[
            "review_count"
        ] = (
            frame[
                "review_count"
            ]
            .astype("int64")
        )

        frame[
            "review_rate_count"
        ] = (
            frame[
                "review_rate_count"
            ]
            .astype("int64")
        )

        return frame
