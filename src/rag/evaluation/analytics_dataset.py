
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_FIELDS = {
    "case_id",
    "split",
    "case_type",
    "question",
}


class AnalyticsEvaluationDataset:
    """
    Fixed, reproducible Manager Analytics benchmark.

    Cases operate on deterministic catalog scopes. Scope validation is
    performed before any LLM call.
    """

    def __init__(
        self,
        cases,
    ):
        self.cases = [
            dict(
                case
            )
            for case
            in cases
        ]

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
                    f"Analytics case {index} is missing: "
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
                    f"Duplicate analytics case_id: {case_id}"
                )

            seen.add(
                case_id
            )

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
                    f"Invalid split for {case_id}: {split}"
                )

            case[
                "case_id"
            ] = case_id

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
                "question"
            ] = str(
                case[
                    "question"
                ]
            ).strip()

            case[
                "filters"
            ] = dict(
                case.get(
                    "filters"
                )
                or {}
            )

            categories = case.get(
                "comparison_categories"
            ) or []

            case[
                "comparison_categories"
            ] = [
                str(
                    value
                ).strip()
                for value
                in categories
                if str(
                    value
                ).strip()
            ]

            case[
                "category_field"
            ] = str(
                case.get(
                    "category_field",
                    "Category2",
                )
            ).strip()

            case[
                "policy_expectations"
            ] = [
                str(
                    value
                ).strip()
                for value
                in (
                    case.get(
                        "policy_expectations"
                    )
                    or []
                )
                if str(
                    value
                ).strip()
            ]

            case[
                "notes"
            ] = str(
                case.get(
                    "notes",
                    "",
                )
            ).strip()

            if (
                case[
                    "comparison_categories"
                ]
                and len(
                    case[
                        "comparison_categories"
                    ]
                )
                not in {
                    2,
                    3,
                }
            ):
                raise ValueError(
                    f"{case_id} comparison must contain "
                    "2 or 3 categories."
                )


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
                    "case_id": case[
                        "case_id"
                    ],
                    "split": case[
                        "split"
                    ],
                    "case_type": case[
                        "case_type"
                    ],
                    "question": case[
                        "question"
                    ],
                    "filters": case[
                        "filters"
                    ],
                    "comparison_categories": (
                        case[
                            "comparison_categories"
                        ]
                    ),
                    "policy_expectations": (
                        case[
                            "policy_expectations"
                        ]
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


    def validate_scopes(
        self,
        analytics_service,
    ):
        errors = []

        for case in self.cases:
            case_id = case[
                "case_id"
            ]

            for field, value in (
                case[
                    "filters"
                ].items()
            ):
                if field not in (
                    analytics_service
                    .products
                    .columns
                ):
                    errors.append(
                        f"{case_id}: unknown filter field {field}"
                    )
                    continue

                valid_values = set(
                    analytics_service
                    .distinct_values(
                        field,
                        include_unknown=True,
                    )
                )

                values = (
                    list(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            list,
                            tuple,
                            set,
                        ),
                    )
                    else [
                        value
                    ]
                )

                for item in values:
                    if str(
                        item
                    ) not in valid_values:
                        errors.append(
                            f"{case_id}: missing {field} value: {item}"
                        )

            categories = case[
                "comparison_categories"
            ]

            if categories:
                field = case[
                    "category_field"
                ]

                if field not in (
                    analytics_service
                    .products
                    .columns
                ):
                    errors.append(
                        f"{case_id}: unknown comparison field {field}"
                    )
                    continue

                valid_values = set(
                    analytics_service
                    .distinct_values(
                        field,
                        include_unknown=True,
                    )
                )

                for category in categories:
                    if category not in (
                        valid_values
                    ):
                        errors.append(
                            f"{case_id}: missing comparison category: "
                            f"{category}"
                        )

        if errors:
            raise ValueError(
                "Analytics benchmark scope validation failed:\n"
                + "\n".join(
                    errors
                )
            )

        return {
            "case_count": len(
                self.cases
            ),
            "dev_count": sum(
                case[
                    "split"
                ]
                == "dev"
                for case
                in self.cases
            ),
            "test_count": sum(
                case[
                    "split"
                ]
                == "test"
                for case
                in self.cases
            ),
            "status": "ok",
        }
