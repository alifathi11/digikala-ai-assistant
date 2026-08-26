# Validated baseline metrics

## Retrieval

| Retriever | Precision@5 | Recall@5 | HitRate@5 | MRR@5 | MAP@5 | nDCG@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 / Tantivy | 0.2093 | 0.8400 | 0.8933 | 0.7203 | 0.6842 | 0.7334 | 83.6 ms | 90.6 ms |
| Embedding / FAISS | 0.2107 | 0.8389 | 0.8933 | 0.6943 | 0.6472 | 0.7082 | 90.9 ms | 98.6 ms |
| Hybrid | **0.2147** | **0.8611** | **0.9267** | **0.7533** | **0.7023** | **0.7558** | 258.1 ms | 269.6 ms |

Hybrid Recall@10 was 0.9722 and HitRate@10 was 0.9933.

## Grounded QA

Samples: **30**

| Metric | Score |
|---|---:|
| Overall | **4.93 / 5** |
| Correctness | 4.90 |
| Relevance | 5.00 |
| Completeness | 4.83 |
| Groundedness | 4.93 |
| Instruction following | 4.97 |
| Safety | 5.00 |
| Citation validity | **100.0%** |
| Evidence precision | 89.2% |
| Evidence recall | 90.6% |
| Evidence F1 | 86.9% |
| Retrieval evidence recall | 92.8% |

Production QA latency:

- Average: **2.75 s**
- P50: **2.70 s**
- P95: **3.68 s**
- Average tokens: **765**

LLM-as-a-Judge overhead:

- Average judge latency: **10.77 s**
- Average judge tokens: **1515**

Cost fields remain `N/A` until exact provider pricing is configured.

## Observed judge failure tags

| Failure tag | Count |
|---|---:|
| ignores_conflict | 4 |
| unsupported_claim | 2 |
| overgeneralization | 1 |
| missed_key_evidence | 1 |
