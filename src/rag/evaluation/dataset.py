import json


class EvaluationDataset:

    REQUIRED_FIELDS = {
        "product_id",
        "query",
        "candidate_ids",
        "relevant_ids"
    }


    def __init__(
        self,
        path,
        validate=True
    ):
        with open(
            path,
            encoding="utf-8"
        ) as f:
            self.samples = json.load(f)

        if validate:
            self.validate()


    def validate(self):
        if not isinstance(
            self.samples,
            list
        ):
            raise ValueError(
                "Evaluation dataset must be a list"
            )

        for idx, sample in enumerate(
            self.samples
        ):
            missing = (
                self.REQUIRED_FIELDS
                - set(sample)
            )

            if missing:
                raise ValueError(
                    f"Sample {idx} is missing: "
                    f"{sorted(missing)}"
                )

            if not isinstance(
                sample["query"],
                str
            ) or not sample["query"].strip():
                raise ValueError(
                    f"Sample {idx} has invalid query"
                )

            candidate_ids = sample[
                "candidate_ids"
            ]

            relevant_ids = sample[
                "relevant_ids"
            ]

            if not candidate_ids:
                raise ValueError(
                    f"Sample {idx} has no candidate_ids"
                )

            if not relevant_ids:
                raise ValueError(
                    f"Sample {idx} has no relevant_ids"
                )

            if not set(relevant_ids).issubset(
                set(candidate_ids)
            ):
                raise ValueError(
                    f"Sample {idx}: relevant_ids must "
                    "be a subset of candidate_ids"
                )

        return True


    def __iter__(self):
        return iter(self.samples)


    def __len__(self):
        return len(self.samples)
