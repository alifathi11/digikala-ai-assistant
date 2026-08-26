# Digikala AI Assistant

A Persian ecommerce assistant built on Digikala product metadata and real user comments.

## Implemented features

- **Grounded Q&A** over product reviews with evidence IDs and hard citation validation.
- **Product Search / Discovery** using product FAISS + Tantivy retrieval, RRF + lexical/category grounding, candidate-scoped review evidence, and grounded LLM reranking.
- **Retrieval, QA, and Product Search evaluation** with reproducible notebooks.
- **Streamlit RTL UI** for Q&A and Product Search.

Product Comparison and Manager Analytics are the next phases and remain disabled in the UI.

## Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ├─────────────────────────────┐
 │                             │
 ▼                             ▼
Grounded QA               Product Search
 │                             │
 ▼                             ▼
Comment Hybrid RAG        Product metadata retrieval
FAISS + Tantivy           FAISS + Tantivy + RRF
 │                             │
 ▼                             ▼
Retrieved reviews         lexical/category grounding
 │                             │
 ▼                             ▼
Grounded LLM              top-12 candidate shortlist
 │                             │
 ▼                             ▼
Citation validation       candidate-scoped review retrieval
                               │
                               ▼
                         Grounded LLM reranker
                               │
                               ▼
                         tiered 30/70 ranking
```

Reusable logic lives under `src/`. Notebooks are orchestration/evaluation only; reusable functions and classes should not be implemented inside notebooks.

## Validated baselines

### Comment retrieval

| Retriever | Recall@5 | MRR@5 | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|
| BM25 / Tantivy | 0.8400 | 0.7203 | 0.7334 | 83.6 ms |
| Embedding / FAISS | 0.8389 | 0.6943 | 0.7082 | 90.9 ms |
| Hybrid | **0.8611** | **0.7533** | **0.7558** | 258.1 ms |

### Grounded QA

- Overall LLM-as-a-Judge: **4.93 / 5**
- Citation validity: **100%**
- Evidence precision: **89.2%**
- Evidence recall: **90.6%**
- Evidence F1: **86.9%**
- Average latency: **2.75 s**
- P95 latency: **3.68 s**

### Product Search

Held-out TEST results on the 30-query LLM-assisted benchmark:

| Metric | Metadata only | LLM-reranked |
|---|---:|---:|
| HitRate@1 | 0.400 | **0.700** |
| MRR@10 | 0.484 | **0.700** |
| nDCG@10 | 0.473 | **0.638** |
| Precision@5 | 0.340 | **0.600** |

All tested LLM-based fusion policies tied. Production keeps **tiered 30% metadata / 70% LLM** so metadata remains a deterministic tie-breaker.

Product Search evaluation latency: **12.10 s mean**, **19.06 s P95**, about **2,564 reranker tokens/query**.

> Product Search qrels are LLM-assisted proxy labels, not independent human gold. Keep this limitation explicit in final reporting.

See `reports/baseline_metrics.md` for the full metric tables.

## Project structure

```text
.
├── app.py
├── configs/
│   ├── rag.yaml
│   ├── qa.yaml
│   ├── qa_evaluation.yaml
│   ├── product_search.yaml
│   ├── product_search_eval_queries.yaml
│   └── product_search_evaluation.yaml
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_build_embedding_index.ipynb
│   ├── 04_build_bm25_index.ipynb
│   ├── 05_retrieval_evaluation.ipynb
│   ├── 06_grounded_qa_demo.ipynb
│   ├── 07_qa_evaluation.ipynb
│   ├── 08_product_search_data_audit.ipynb
│   ├── 09_build_product_search_indexes.ipynb
│   ├── 10_product_search_smoke_test.ipynb
│   ├── 11_product_search_metadata_debug.ipynb
│   ├── 12_product_search_build_qrels.ipynb
│   └── 13_product_search_evaluation.ipynb
├── reports/
├── scripts/
├── src/
│   ├── app/
│   └── rag/
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `METIS_API_KEY` and `METIS_BASE_URL` in `.env`.

Expected local data layout:

```text
data/
├── raw/
├── processed/
├── indexes/
└── evaluation/
```

## Rebuild / evaluation order

Run notebooks in numeric order:

1. `01`–`04`: preprocessing and comment indexes
2. `05`: retrieval evaluation
3. `06`–`07`: grounded QA and QA evaluation
4. `08`–`09`: Product Search audit and indexes
5. `10`–`11`: Product Search smoke/debug
6. `12`: Product Search qrels
7. `13`: Product Search held-out evaluation

Product Search qrel/evaluation notebooks use resume/cache to avoid repeating completed LLM calls.

## Run the UI

```bash
streamlit run app.py
```

## Project hygiene

Generated caches, Jupyter checkpoints, local data/indexes and `.env` are git-ignored. The abandoned BERT/NLI stance-calibration experiment is not part of the active project.

See `CLEANUP.md` for the cleanup performed before the Product Comparison phase.
