# Product Search Evaluation

Benchmark:
- Queries: **30**
- Query types: brand, attribute, experiential, negative, multi-constraint
- Qrels: **587**
- Queries with no relevant product in judged pool: **8**
- Production ranking policy: **tiered_30_70**

## Held-out TEST Results

| Metric | Metadata Only | tiered_30_70 |
|---|---:|---:|
| MRR@10 | 0.4843 | **0.7000** |
| Precision@1 | 0.4000 | **0.7000** |
| HitRate@1 | 0.4000 | **0.7000** |
| nDCG@1 | 0.3429 | **0.7000** |
| Precision@3 | 0.3000 | **0.6333** |
| nDCG@3 | 0.3382 | **0.6765** |
| Precision@5 | 0.3400 | **0.5800** |
| nDCG@5 | 0.3742 | **0.6583** |
| Precision@10 | 0.3600 | **0.3800** |
| nDCG@10 | 0.4801 | **0.6352** |

Candidate retrieval:
- DEV Candidate Recall@12: 0.5076
- DEV Candidate Hit@12: 0.6500
- TEST Candidate Recall@12: **0.6275**
- TEST Candidate Hit@12: **0.7000**

Failures across 30 queries:
- `ok`: 19
- `no_relevant_in_judged_pool`: 8
- `candidate_retrieval_miss`: 2
- `top_rank_error`: 1

Performance:
- Mean search latency: **14.71 s**
- P95 search latency: **23.08 s**
- Total reranker tokens: **77,432**
- Reranker API cost: **$0.1944**
