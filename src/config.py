import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss" / "index.bin"
FAISS_META_PATH = BASE_DIR / "data" / "faiss" / "meta.pkl"

PRODUCTS_CSV = DATA_RAW_DIR / "digikala-products.csv"
COMMENTS_CSV = DATA_RAW_DIR / "digikala-comments.csv"
PROCESSED_COMMENTS_CSV = DATA_PROCESSED_DIR / "comments_clean.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 64
TOP_K = 5

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

MIN_COMMENT_LENGTH = 10
