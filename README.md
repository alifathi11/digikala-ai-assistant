# Digikala AI Assistant

A Persian ecommerce AI assistant built on Digikala product metadata and real user reviews. The project combines deterministic retrieval/analytics with grounded LLM generation and explicit evaluation.

## Final features

### 1. Grounded Product Q&A

- Hybrid comment retrieval with **Tantivy BM25 + FAISS embeddings**.
- Product-scoped question answering over retrieved reviews.
- Structured answers with evidence IDs, confidence, and insufficient-evidence handling.
- Hard citation validation, repair retry, and deterministic sanitization.

### 2. Product Search / Discovery

- Canonical product-level metadata index.
- Product FAISS + Tantivy retrieval.
- Weighted RRF plus title/category lexical grounding and exact-brand boost.
- Candidate-scoped review evidence retrieval.
- Grounded LLM reranking with `support / mixed / contradict / none` evidence status.
- Frozen production ranking policy: **`tiered_30_70`**.

### 3. Product Comparison

- Compare two or three selected products.
- Direct metadata plus separately retrieved review evidence for every product.
- Criterion-by-criterion grounded comparison.
- Strict citation ownership validation so evidence cannot be assigned to the wrong product.
- Deterministic winner checks for metadata-only criteria and calibrated no-winner behavior.

### 4. Manager Analytics

- Canonical product-level aggregation with batched review statistics.
- Category overview and category comparison.
- Price, rating, rating coverage, review coverage, brand, and engagement analytics.
- Data-quality guards for limited brand coverage, sparse historical price, and truncated review-volume signals.
- Manager Q&A where the LLM cannot invent numbers: it can only reference verified metric placeholders rendered from Python-computed facts.

## Streamlit UI

Run:

```bash
streamlit run app.py
```

The UI includes:

- Grounded Q&A
- Product Search
- Product Comparison
- Manager Analytics

## Architecture

```text
Raw Digikala data
        |
        v
Preprocessing + canonicalization
        |
        +-------------------------+
        |                         |
        v                         v
Comment indexes              Product indexes
FAISS + Tantivy              FAISS + Tantivy
        |                         |
        v                         v
Grounded QA                 Product Search
                                  |
                                  v
                          Grounded LLM reranker
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
             Product Comparison          Manager Analytics
             metadata + reviews          deterministic metrics
                     |                         |
                     v                         v
             citation ownership        verified metric placeholders
```

Reusable logic is implemented under `src/`. Notebooks contain only orchestration, evaluation, and short English notes.

## Central configuration

The project has a single configuration file:

```text
configs/project.yaml
```

It contains retrieval, generation, Product Search, Product Comparison, Manager Analytics, benchmark cases, evaluation settings, and model pricing.

### Token pricing

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---:|---:|
| `gpt-5.6-terra` | **$1** | **$6** |
| `gpt-5.6-sol` | **$5** | **$30** |

Current usage:

- Production generation/reranking: `gpt-5.6-terra`
- QA qualitative judge: `gpt-5.6-sol`
- Product Search relevance teacher, Product Comparison judge, and Manager Analytics judge: `gpt-5.6-terra`

Evaluation notebooks record prompt/completion tokens and estimated USD cost using these prices.

## Final evaluation summary

| Section | Main held-out result | Grounding / deterministic result |
|---|---|---|
| Grounded QA | **4.93 / 5** overall | **100% citation validity** |
| Product Search | **0.638 nDCG@10**, **0.70 HitRate@1** on TEST | LLM reranking vs 0.473 nDCG@10 metadata-only |
| Product Comparison | **4.88 / 5** on TEST | **100% citation ownership and deterministic winner accuracy** |
| Manager Analytics | **4.995 / 5** on TEST | **100% numeric faithfulness and fact accuracy** |

Detailed validated results are stored in `reports/`:

```text
reports/
├── 01_grounded_qa_evaluation.md
├── 02_product_search_evaluation.md
├── 03_product_comparison_evaluation.md
└── 04_manager_analytics_evaluation.md
```

Important evaluation limitations are documented inside each report. Product Search qrels and qualitative judges are LLM-assisted proxies rather than independent human gold.

## Final notebooks

Run notebooks in order:

```text
01_data_analysis.ipynb
02_data_preprocessing.ipynb
03_build_comment_embedding_index.ipynb
04_build_comment_bm25_index.ipynb
05_retrieval_evaluation.ipynb
06_grounded_qa_evaluation.ipynb
07_build_product_search_indexes.ipynb
08_build_product_search_qrels.ipynb
09_product_search_evaluation.ipynb
10_product_comparison_evaluation.ipynb
11_manager_analytics_data_audit.ipynb
12_manager_analytics_evaluation.ipynb
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set:

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

## Project structure

```text
.
├── app.py
├── configs/
│   └── project.yaml
├── notebooks/
│   └── 01 ... 12
├── reports/
│   └── four final evaluation reports
├── scripts/
│   └── generate_eval_dataset.py
├── src/
│   ├── app/
│   └── rag/
├── tests/
├── .env.example
├── .gitignore
├── pytest.ini
└── requirements.txt
```

## Testing

```bash
pytest -q
```
