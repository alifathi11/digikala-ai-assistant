from pathlib import Path

import pandas as pd
import yaml


REQUIRED_FIELDS = {
    "case_id",
    "split",
    "case_type",
    "product_ids",
    "query",
}


class ComparisonEvaluationDataset:
    """Fixed, reproducible benchmark cases for Product Comparison."""

    def __init__(
        self,
        cases,
    ):
        self.cases = list(
            cases
        )

        self._validate()


    def _validate(
        self,
    ):
        seen = set()

        for index, case in enumerate(
            self.cases
        ):
            missing = (
                REQUIRED_FIELDS
                - set(
                    case
                )
            )

            if missing:
                raise ValueError(
                    "Comparison case "
                    f"{index} is missing: "
                    f"{sorted(missing)}"
                )

            case_id = str(
                case[
                    "case_id"
                ]
            ).strip()

            if not case_id:
                raise ValueError(
                    "case_id cannot be empty."
                )

            if case_id in seen:
                raise ValueError(
                    "Duplicate comparison "
                    f"case_id: {case_id}"
                )

            seen.add(
                case_id
            )

            product_ids = []
            product_seen = set()

            for value in case[
                "product_ids"
            ]:
                product_id = int(
                    value
                )

                if product_id in (
                    product_seen
                ):
                    continue

                product_seen.add(
                    product_id
                )
                product_ids.append(
                    product_id
                )

            if len(product_ids) not in {
                2,
                3,
            }:
                raise ValueError(
                    f"{case_id} must contain "
                    "2 or 3 unique product IDs."
                )

            case[
                "product_ids"
            ] = product_ids

            split = str(
                case[
                    "split"
                ]
            ).strip().lower()

            if split not in {
                "dev",
                "test",
            }:
                raise ValueError(
                    f"Invalid split for {case_id}: "
                    f"{split}"
                )

            case[
                "split"
            ] = split

            case[
                "case_type"
            ] = str(
                case[
                    "case_type"
                ]
            ).strip()

            case[
                "query"
            ] = str(
                case[
                    "query"
                ]
            ).strip()


    @classmethod
    def load(
        cls,
        path,
    ):
        path = Path(
            path
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = yaml.safe_load(
                handle
            )

        return cls(
            payload.get(
                "cases",
                [],
            )
        )


    def to_frame(
        self,
    ):
        rows = []

        for case in self.cases:
            rows.append(
                {
                    **case,
                    "product_count": len(
                        case[
                            "product_ids"
                        ]
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


    def validate_products(
        self,
        product_documents,
    ):
        available = set(
            pd.to_numeric(
                product_documents[
                    "id"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        missing = []

        for case in self.cases:
            absent = [
                product_id
                for product_id
                in case[
                    "product_ids"
                ]
                if product_id
                not in available
            ]

            if absent:
                missing.append(
                    {
                        "case_id": case[
                            "case_id"
                        ],
                        "missing_product_ids": (
                            absent
                        ),
                    }
                )

        return missing
