# Phase 9c -- CONSOLIDATED Failure Classification

**Scope note:** each record's expected (law, section) answer is looked up independently in the semantic-only ranked list, the BM25-only ranked list, and the fused hybrid list (alpha=0.5). Classification logic:

- `FUSION_HURTS`: one individual retriever alone ranked the expected answer #1, but fusion did not -- actionable via alpha, not a new model.

- `GENUINE_MISS`: expected answer absent from BOTH individual retrievers' top-30 entirely -- a real representation/retrieval gap.

- `RANKING_ONLY`: expected answer is in the hybrid system's top-3, just not top-1 -- recall is fine, this is what a reranker should fix.

- `CORRECT`: hybrid top-1 already matches (not a failure).


## Per-record results

| ID | Expected | Semantic rank | BM25 rank | Hybrid rank | Hybrid top-1 | Diagnosis |
|---|---|---|---|---|---|---|
| EV1020 | BNS 2(28) | 1 | 6 | 2 | BNS 255 | FUSION_HURTS |
| EV1023 | BNS 5 | 2 | 1 | 1 | BNS 5 | CORRECT |
| EV1033 | BNS 120 | 1 | 1 | 1 | BNS 120 | CORRECT |
| EV1043 | BNS 127 | 7 | None | 16 | BNS 126 | GENUINE_MISS |
| EV1048 | BNS 127 | 1 | 4 | 2 | BNS 258 | FUSION_HURTS |
| EV1068 | BNS 179 | 1 | 17 | 5 | BNS 180 | FUSION_HURTS |
| EV1082 | BNS 181 | 1 | 1 | 1 | BNS 181 | CORRECT |
| EV1090 | BNS 309 | 3 | 8 | 1 | BNS 309 | CORRECT |
| EV1091 | BNS 309 | 2 | 7 | 2 | BNS 312 | RANKING_ONLY |
| EV1118 | BNS 340 | 1 | 1 | 1 | BNS 340 | CORRECT |

## Diagnosis breakdown

- **FUSION_HURTS**: 3
- **CORRECT**: 5
- **GENUINE_MISS**: 1
- **RANKING_ONLY**: 1