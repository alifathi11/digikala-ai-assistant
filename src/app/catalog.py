from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class ProductCatalog:
    DISPLAY_COLUMNS = (
        "id",
        "title_fa",
        "Brand",
        "Category1",
        "Category2",
        "Rate",
        "Rate_cnt",
        "Price",
    )


    def __init__(
        self,
        products,
    ):
        if "id" not in products:
            raise ValueError(
                "Product catalog requires "
                "an id column."
            )

        self.products = (
            products
            .drop_duplicates(
                subset="id",
                keep="first",
            )
            .reset_index(drop=True)
            .copy()
        )

        self._rows = {
            int(row.id): row
            for row in (
                self.products
                .itertuples(
                    index=False
                )
            )
        }


    @classmethod
    def from_parquet(
        cls,
        path,
        allowed_product_ids=None,
    ):
        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Products parquet "
                f"not found: {path}"
            )

        available_columns = (
            pq.read_schema(
                path
            ).names
        )

        columns = [
            column
            for column
            in cls.DISPLAY_COLUMNS
            if column
            in available_columns
        ]

        products = pd.read_parquet(
            path,
            columns=columns,
        )

        if (
            allowed_product_ids
            is not None
        ):
            allowed = {
                int(value)
                for value
                in allowed_product_ids
            }

            products = (
                products[
                    products["id"]
                    .astype(int)
                    .isin(allowed)
                ]
                .copy()
            )

        return cls(
            products
        )


    @property
    def product_ids(
        self,
    ):
        return [
            int(value)
            for value
            in self.products[
                "id"
            ].tolist()
        ]


    def get(
        self,
        product_id,
    ):
        return self._rows.get(
            int(product_id)
        )


    def label(
        self,
        product_id,
    ):
        row = self.get(
            product_id
        )

        if row is None:
            return (
                f"محصول {product_id}"
            )

        title = getattr(
            row,
            "title_fa",
            None,
        )

        title = (
            str(title).strip()
            if title is not None
            else ""
        )

        if not title:
            title = "بدون عنوان"

        return (
            f"{title} · "
            f"#{int(product_id)}"
        )
