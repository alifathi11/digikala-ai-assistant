from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
)
from sklearn.pipeline import (
    Pipeline,
)
from sklearn.preprocessing import (
    StandardScaler,
)


LABELS = (
    "support",
    "neutral",
    "contradict",
)

PROBABILITY_COLUMNS = (
    "support_prob",
    "neutral_prob",
    "contradict_prob",
)


def _feature_matrix(
    frame,
):
    """
    Log-probability features are more suitable than raw probabilities for
    correcting systematic class bias in the base NLI model.
    """
    probabilities = (
        frame[
            list(
                PROBABILITY_COLUMNS
            )
        ]
        .astype(float)
        .to_numpy()
    )

    probabilities = np.clip(
        probabilities,
        1e-6,
        1.0,
    )

    return np.log(
        probabilities
    )


def collect_stance_predictions(
    classifier,
    dataset,
):
    """
    Run the base classifier grouped by query so each pair receives the exact
    same NLI formulation used in production.
    """
    rows = []

    for query, group in (
        dataset
        .groupby(
            "query",
            sort=False,
        )
    ):
        prediction = (
            classifier
            .predict(
                query=query,
                reviews=(
                    group[
                        "review_text"
                    ]
                    .fillna("")
                    .astype(str)
                    .tolist()
                ),
                review_ids=(
                    group[
                        "review_id"
                    ]
                    .astype(int)
                    .tolist()
                ),
            )
        )

        prediction[
            "query"
        ] = query

        rows.append(
            prediction
        )

    predictions = (
        pd.concat(
            rows,
            ignore_index=True,
        )
        if rows
        else pd.DataFrame()
    )

    return (
        dataset
        .merge(
            predictions[
                [
                    "query",
                    "review_id",
                    "support_prob",
                    "neutral_prob",
                    "contradict_prob",
                    "stance_score",
                    "stance_label",
                ]
            ],
            on=[
                "query",
                "review_id",
            ],
            how="left",
        )
        .rename(
            columns={
                "stance_label": (
                    "base_predicted_label"
                ),
            }
        )
    )


def classification_summary(
    y_true,
    y_pred,
):
    y_true = [
        str(value)
        .strip()
        .lower()
        for value
        in y_true
    ]

    y_pred = [
        str(value)
        .strip()
        .lower()
        for value
        in y_pred
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            LABELS
        ),
    )

    per_class = {}

    for label_index, label in (
        enumerate(
            LABELS
        )
    ):
        true_positive = (
            matrix[
                label_index,
                label_index,
            ]
        )

        predicted_total = (
            matrix[
                :,
                label_index,
            ].sum()
        )

        true_total = (
            matrix[
                label_index,
                :,
            ].sum()
        )

        precision = (
            true_positive
            / predicted_total
            if predicted_total
            else 0.0
        )

        recall = (
            true_positive
            / true_total
            if true_total
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        per_class[
            label
        ] = {
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "support": int(
                true_total
            ),
        }

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(
                    LABELS
                ),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": (
            per_class
        ),
        "confusion_matrix": {
            "labels": list(
                LABELS
            ),
            "values": (
                matrix
                .astype(int)
                .tolist()
            ),
        },
    }


class StanceProbabilityCalibrator:

    def __init__(
        self,
        c_values=(
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ),
        max_cv_splits=5,
        random_state=42,
    ):
        self.c_values = tuple(
            float(value)
            for value
            in c_values
        )

        self.max_cv_splits = int(
            max_cv_splits
        )

        self.random_state = int(
            random_state
        )

        self.model = None
        self.best_c = None
        self.cv_results = None


    def _new_model(
        self,
        c_value,
    ):
        return Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "logistic",
                    LogisticRegression(
                        C=float(
                            c_value
                        ),
                        class_weight=(
                            "balanced"
                        ),
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=(
                            self.random_state
                        ),
                    ),
                ),
            ]
        )


    def fit(
        self,
        frame,
        label_column="label",
        group_column="query_id",
    ):
        frame = (
            frame
            .copy()
            .reset_index(drop=True)
        )

        x = _feature_matrix(
            frame
        )

        y = (
            frame[
                label_column
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_numpy()
        )

        groups = (
            frame[
                group_column
            ]
            .astype(str)
            .to_numpy()
        )

        unique_groups = np.unique(
            groups
        )

        n_splits = min(
            self.max_cv_splits,
            len(
                unique_groups
            ),
        )

        if n_splits < 2:
            raise ValueError(
                "Calibration requires at "
                "least two query groups."
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        cv_rows = []

        for c_value in (
            self.c_values
        ):
            fold_scores = []

            for (
                train_indices,
                valid_indices,
            ) in splitter.split(
                x,
                y,
                groups=groups,
            ):
                model = self._new_model(
                    c_value
                )

                model.fit(
                    x[
                        train_indices
                    ],
                    y[
                        train_indices
                    ],
                )

                predicted = (
                    model.predict(
                        x[
                            valid_indices
                        ]
                    )
                )

                score = f1_score(
                    y[
                        valid_indices
                    ],
                    predicted,
                    labels=list(
                        LABELS
                    ),
                    average="macro",
                    zero_division=0,
                )

                fold_scores.append(
                    float(
                        score
                    )
                )

            cv_rows.append(
                {
                    "C": float(
                        c_value
                    ),
                    "macro_f1_mean": float(
                        np.mean(
                            fold_scores
                        )
                    ),
                    "macro_f1_std": float(
                        np.std(
                            fold_scores
                        )
                    ),
                    "folds": int(
                        len(
                            fold_scores
                        )
                    ),
                }
            )

        self.cv_results = (
            pd.DataFrame(
                cv_rows
            )
            .sort_values(
                [
                    "macro_f1_mean",
                    "macro_f1_std",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        self.best_c = float(
            self.cv_results
            .iloc[0][
                "C"
            ]
        )

        self.model = (
            self._new_model(
                self.best_c
            )
        )

        self.model.fit(
            x,
            y,
        )

        return self


    def predict(
        self,
        frame,
    ):
        if self.model is None:
            raise RuntimeError(
                "Calibrator is not fitted."
            )

        x = _feature_matrix(
            frame
        )

        predicted = (
            self.model.predict(
                x
            )
        )

        probabilities = (
            self.model
            .predict_proba(
                x
            )
        )

        classes = (
            self.model
            .named_steps[
                "logistic"
            ]
            .classes_
        )

        output = frame.copy()

        output[
            "calibrated_label"
        ] = predicted

        for label in (
            LABELS
        ):
            if label in classes:
                class_index = int(
                    np.where(
                        classes
                        == label
                    )[0][0]
                )

                output[
                    f"calibrated_{label}_prob"
                ] = probabilities[
                    :,
                    class_index,
                ]
            else:
                output[
                    f"calibrated_{label}_prob"
                ] = 0.0

        return output


    def save(
        self,
        path,
    ):
        if self.model is None:
            raise RuntimeError(
                "Calibrator is not fitted."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model": self.model,
            "best_c": (
                self.best_c
            ),
            "labels": (
                LABELS
            ),
            "probability_columns": (
                PROBABILITY_COLUMNS
            ),
        }

        joblib.dump(
            payload,
            path,
        )

        if self.cv_results is not None:
            self.cv_results.to_csv(
                path.with_suffix(
                    ".cv.csv"
                ),
                index=False,
            )

        return path


    @classmethod
    def load(
        cls,
        path,
    ):
        payload = joblib.load(
            path
        )

        instance = cls()

        instance.model = (
            payload[
                "model"
            ]
        )

        instance.best_c = (
            payload.get(
                "best_c"
            )
        )

        return instance
