# Grounded QA Evaluation

## Retrieval

| Metric | BM25 | Embedding | Hybrid |
|---|---:|---:|---:|
| Precision@1 | 0.6467 | 0.6067 | **0.6933** |
| Recall@1 | 0.5522 | 0.5089 | **0.5922** |
| Recall@5 | 0.8500 | 0.8522 | **0.8900** |
| MRR@5 | 0.7422 | 0.7220 | **0.7850** |
| nDCG@5 | 0.7499 | 0.7280 | **0.7883** |
| Recall@10 | 0.9478 | 0.9378 | **0.9611** |
| MRR@10 | 0.7548 | 0.7299 | **0.7918** |
| nDCG@10 | 0.7847 | 0.7601 | **0.8149** |
| Mean Latency | 90.87 ms | 99.57 ms | 263.82 ms |
| P95 Latency | 109.27 ms | 123.90 ms | 278.53 ms |

Retrieval benchmark: 150 queries from 50 products.

Benchmark query generation:
- Prompt tokens: 45,607
- Completion tokens: 9,661
- Total tokens: 55,268
- API cost: **$0.1036**

## Grounded QA

Evaluation samples: **30**

| Metric | Result |
|---|---:|
| Overall Score | **4.94 / 5** |
| Correctness | 4.93 |
| Groundedness | 4.97 |
| Relevance | 4.97 |
| Completeness | 4.83 |
| Instruction Following | 5.00 |
| Safety | 5.00 |
| Citation Validity | **100.0%** |
| Evidence Precision | 80.0% |
| Evidence Recall | 87.8% |
| Evidence F1 | 81.6% |
| Retrieval Evidence Recall | 90.0% |

Performance:
- Average QA latency: **3.00 s**
- P50 QA latency: 2.77 s
- P95 QA latency: 4.81 s
- Average QA tokens: 762
- QA API cost: **$0.0341**
- Average judge latency: 8.62 s
- Average judge tokens: 1,481
- Judge API cost: **$0.5344**
- Total QA evaluation API cost: **$0.5685**

Failures:
- `missed_key_evidence`: 3 cases
