Embedder: `st` | LLM: `extractive` | chunks: 15 | questions: 19

| Retrieval strategy | Hit@3 | MRR | Avg latency (ms) |
|---|---|---|---|
| bm25 | 0.94 | 0.94 | 8.89 |
| dense | 1.00 | 1.00 | 9.43 |
| hybrid (RRF) | 1.00 | 0.97 | 9.77 |
| hybrid + rerank | 0.94 | 0.94 | 9.44 |

| End-to-end metric | Score |
|---|---|
| Answer rate (answerable Qs) | 0.88 |
| Grounded answers | 1.00 |
| Refusal accuracy (unanswerable Qs) | 1.00 |

Semantic cache: cold 26.71 ms -> warm 11.65 ms avg per request
