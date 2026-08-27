# Manager Analytics Evaluation

## Data Audit

- Canonical products: **948,352**
- Unique product IDs: 948,352
- Duplicate product IDs: 0
- Comment rows: **6,153,060**
- Comment-to-product join rate: **100%**
- Products with reviews: **331,599**
- Product review coverage: **34.97%**
- Current price coverage: **99.98%**
- Rated-product coverage: **38.06%**
- Category1 usable coverage: **100%**
- Category2 usable coverage: **80.88%**
- Brand usable coverage: **44.11%**
- Historical-price coverage: **5.86%**
- Valid review-rating coverage: **91.32%**

Metric readiness:
- Ready: product count, current price, Category1, Category2, sub-category, review presence/coverage, review-volume ranking, review-rating statistics
- Limited: product-rating statistics, brand analysis
- Unavailable: historical-price statistics

## Manager Analytics Evaluation

Benchmark:
- Cases: **15**
- DEV: 5
- TEST: 10
- Successful executions: **15 / 15**

| Metric | DEV | TEST | ALL |
|---|---:|---:|---:|
| Overall Judge Score | 4.83 | **4.86** | **4.85** |
| Numeric Faithfulness | 100% | **100%** | **100%** |
| Fact Value Accuracy | 100% | **100%** | **100%** |
| Scope Product Count Accuracy | 100% | **100%** | **100%** |
| Comparison Fact Accuracy | 100% | **100%** | **100%** |
| Rendered Metric Accuracy | 100% | **100%** | **100%** |
| Policy Guard Configuration | 100% | **100%** | **100%** |
| Relevance | 5.00 | **5.00** | 5.00 |
| Completeness | 4.60 | **4.70** | 4.67 |
| Managerial Usefulness | 4.60 | **4.90** | 4.80 |
| Instruction Following | 5.00 | **4.90** | 4.93 |

Failures:
- `missed_requested_metric`: 3
- `unsupported_claim`: 2
- Flagged cases: `a004`, `a012`, `a014`

Performance:
- Mean answer latency: **14.29 s**
- P95 answer latency: **23.60 s**
- Mean judge latency: **11.16 s**
- Mean end-to-end evaluation latency: **26.00 s**
- Mean answer tokens: 3,427.53
- Mean judge tokens: 4,335.33
- Answer API cost: **$0.0952**
- Judge API cost: **$0.1114**
- Total evaluation API cost: **$0.2066**
