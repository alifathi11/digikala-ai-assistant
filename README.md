# Digikala AI Assistant

A Persian ecommerce assistant built on Digikala product comments.

The current baseline supports:

- dense comment retrieval with **FAISS**
- sparse BM25 retrieval with **Tantivy**
- **Hybrid** retrieval
- grounded product Q&A over real user comments
- evidence/citation IDs with hard validation
- retrieval evaluation
- QA evaluation with deterministic metrics and **LLM-as-a-Judge**
- a Streamlit UI designed to grow into product search, comparison and analytics

## Current validated baseline

### Retrieval benchmark

| Retriever | Recall@5 | MRR@5 | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|
| BM25 / Tantivy | 0.8400 | 0.7203 | 0.7334 | 83.6 ms |
| Embedding / FAISS | 0.8389 | 0.6943 | 0.7082 | 90.9 ms |
| Hybrid | **0.8611** | **0.7533** | **0.7558** | 258.1 ms |

### Grounded QA benchmark

- Overall LLM-as-a-Judge score: **4.93 / 5**
- Correctness: **4.90**
- Relevance: **5.00**
- Completeness: **4.83**
- Groundedness: **4.93**
- Instruction following: **4.97**
- Safety: **5.00**
- Citation validity: **100%**
- Evidence precision: **89.2%**
- Evidence recall: **90.6%**
- Average QA latency: **2.75 s**
- P95 QA latency: **3.68 s**

Judge latency is evaluation overhead and is intentionally reported separately
from production QA latency.

## Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
GroundedQAPipeline
 │
 ├───────────────┐
 ▼               ▼
Tantivy BM25    FAISS embeddings
 │               │
 └───────┬───────┘
         ▼
   Hybrid fusion
         │
         ▼
 Retrieved comments
         │
         ▼
 OpenAI-compatible LLM
         │
         ▼
 Answer + evidence IDs
         │
         ▼
 Citation validation / repair
```

Reusable logic is kept under `src/`. Notebooks and UI pages only orchestrate
those modules.

## Project structure

```text
.
├── app.py
├── configs/
│   ├── rag.yaml
│   ├── qa.yaml
│   └── qa_evaluation.yaml
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_build_embedding_index.ipynb
│   ├── 04_build_bm25_index.ipynb
│   ├── 05_retrieval_evaluation.ipynb
│   ├── 06_grounded_qa_demo.ipynb
│   └── 07_qa_evaluation.ipynb
├── scripts/
│   └── generate_eval_dataset.py
├── src/
│   ├── app/
│   └── rag/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── .env.example
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```text
METIS_API_KEY=...
METIS_BASE_URL=...
```

Expected local data layout:

```text
data/
├── raw/
│   ├── digikala-products.csv
│   └── digikala-comments.csv
├── processed/
├── indexes/
└── evaluation/
```

## Rebuild order

Run the notebooks in order:

1. `01_data_analysis.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_build_embedding_index.ipynb`
4. `04_build_bm25_index.ipynb`

Generate the retrieval benchmark if needed:

```bash
python scripts/generate_eval_dataset.py
```

Then run:

5. `05_retrieval_evaluation.ipynb`
6. `06_grounded_qa_demo.ipynb`
7. `07_qa_evaluation.ipynb`

## Run the UI

After processed files and indexes exist:

```bash
streamlit run app.py
```

The current UI exposes grounded Q&A. The sidebar already reserves the feature
layout for the next project stages:

- Product Search
- Product Comparison
- Manager Analytics

New UI features should be added as pages/components under `src/app/`; they
should call reusable services/pipelines in `src/` rather than implementing
retrieval or generation logic inside the UI.

## Retrieval implementation

### Embeddings

Model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The comment corpus is encoded incrementally and stored as chunked
`IndexFlatIP` FAISS files. Embeddings and queries are L2-normalized.

### BM25

Tantivy stores a single global disk-backed inverted index. This replaced the
old `rank_bm25` chunk scan, removing the multi-second query bottleneck and
ensuring global BM25 statistics.

### Hybrid

Hybrid retrieval normalizes and fuses BM25 and embedding scores using weights
from `configs/rag.yaml`.

## Evaluation scope

The retrieval benchmark uses annotated candidate pools. QA evaluation uses the
same candidate pools so evidence precision/recall is measured only against
fully annotated candidates.

Production product Q&A is not candidate-limited: it searches comments belonging
to the selected product.

See `CLEANUP.md` for the cleanup rationale and removed legacy files.
