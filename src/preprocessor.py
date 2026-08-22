import re
import pickle
import pandas as pd
from pathlib import Path
from src.config import (
    COMMENTS_CSV,
    PROCESSED_COMMENTS_CSV,
    DATA_PROCESSED_DIR,
    FAISS_META_PATH,
    MIN_COMMENT_LENGTH,
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].apply(clean_text)
    df = df[df["body"].str.len() >= MIN_COMMENT_LENGTH]
    df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)
    return df


def run():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path(FAISS_META_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(COMMENTS_CSV, low_memory=False)
    df = preprocess_comments(df)
    df.to_csv(PROCESSED_COMMENTS_CSV, index=False)
    print(f"Saved {len(df)} clean comments to {PROCESSED_COMMENTS_CSV}")
