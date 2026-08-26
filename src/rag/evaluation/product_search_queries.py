from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_product_search_queries(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    frame = pd.DataFrame(payload.get("queries", []))
    required = {"query_id", "query_type", "query"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Query bank is missing columns: {sorted(missing)}")

    frame = frame.copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()

    if frame["query_id"].duplicated().any():
        raise ValueError("query_id values must be unique.")

    return frame


def assign_stratified_split(queries, test_fraction=0.25, seed=42):
    frame = queries.copy().reset_index(drop=True)
    frame["split"] = "dev"
    rng = np.random.default_rng(int(seed))

    for _, group in frame.groupby("query_type", sort=True):
        indexes = rng.permutation(group.index.to_numpy(copy=True))
        if len(indexes) <= 1:
            continue
        n_test = max(1, int(round(len(indexes) * float(test_fraction))))
        n_test = min(n_test, len(indexes) - 1)
        frame.loc[indexes[:n_test], "split"] = "test"

    return frame
