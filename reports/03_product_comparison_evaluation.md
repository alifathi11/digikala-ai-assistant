# Product Comparison Evaluation

Benchmark:
- Cases: **15**
- DEV: 5
- TEST: 10
- Successful executions: **15 / 15**

## Final Scores

| Metric | DEV | TEST | ALL |
|---|---:|---:|---:|
| Overall Score | 4.860 | **4.985** | **4.943** |
| Correctness | 4.80 | **5.00** | 4.93 |
| Groundedness | 4.80 | **5.00** | 4.93 |
| Criterion Coverage | 5.00 | **5.00** | 5.00 |
| Conflict Handling | 5.00 | **5.00** | 5.00 |
| Recommendation Calibration | 4.60 | **4.90** | 4.80 |
| Relevance | 5.00 | **5.00** | 5.00 |
| Instruction Following | 5.00 | **5.00** | 5.00 |
| Safety | 5.00 | **5.00** | 5.00 |

Deterministic checks:
- Citation Validity: **100%**
- Citation Ownership: **100%**
- Assessment Product Coverage: **100%**
- Deterministic Winner Accuracy: **100%**
- TEST No-winner Accuracy: **100%**

Failure:
- `insufficient_evidence_mishandled`: 1 DEV case (`c005`)

Insufficient-evidence stress test:
- `c015`: no winner returned
- `insufficient_evidence=True`
- No-winner accuracy: **100%**

Performance:
- Mean comparison latency: **11.08 s**
- P95 comparison latency: 17.51 s
- Mean end-to-end evaluation latency: **21.10 s**
- P95 end-to-end evaluation latency: 30.66 s
- Mean comparison tokens: 1,635.67
- Mean judge tokens: 2,518.07
- Mean end-to-end tokens: 4,153.73
- Comparison API cost: **$0.0727**
- Judge API cost: **$0.0744**
- Total evaluation API cost: **$0.1472**
