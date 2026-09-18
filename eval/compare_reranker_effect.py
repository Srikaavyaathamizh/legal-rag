from __future__ import annotations

import sys
import os
import json
from datetime import date as _date

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

import bm25_index as bm25_mod
import semantic_index as sem_mod

from hybrid_ranker import fuse_results
from section_reranker import rerank_sections


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "unified_sections.json",
)

EVAL_PATH = os.path.join(
    os.path.dirname(__file__),
    "eval_dataset.json",
)

ALPHA = 0.3
RETRIEVAL_K = 30
TOP_K = 10


def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def key(record):
    return (
        record["law"],
        record["section"],
    )


def rank_keys(results):
    return [
        key(r.section_record)
        for r in results
    ]


def rank_of(ranked, expected):
    for i, item in enumerate(ranked, start=1):
        if item == expected:
            return i
    return None


def main():

    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    eval_records = [
        r for r in eval_records
        if r["category"] == "CONSOLIDATED"
    ]

    print("=" * 90)
    print("PHASE 9f — RERANKER EFFECT ANALYSIS")
    print("=" * 90)

    print(f"CONSOLIDATED records: {len(eval_records)}")
    print(f"Alpha: {ALPHA}")
    print(f"Retrieval K: {RETRIEVAL_K}")
    print(f"Final K: {TOP_K}")

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("Loading BM25")
    print("=" * 90)

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    # ---------------------------------------------------------
    # Semantic
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("Loading REAL SentenceTransformer + Qdrant")
    print("=" * 90)

    semantic_idx = sem_mod.SemanticIndex()
    semantic_idx.build(sections)

    print(
        "Semantic backend:",
        type(semantic_idx).__name__
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    fusion_hit1 = 0
    fusion_hit3 = 0

    rerank_hit1 = 0
    rerank_hit3 = 0

    print()
    print("=" * 90)
    print("COMPARISON")
    print("=" * 90)

    for rec in eval_records:

        expected = (
            rec["expected_law"],
            rec["expected_section"],
        )

        incident_date = parse_date(
            rec["incident_date"]
        )

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        bm25_results, _ = (
            bm25_mod.search_with_temporal_filter(
                bm25_idx,
                sections,
                rec["query"],
                incident_date=incident_date,
                top_k=RETRIEVAL_K,
            )
        )

        # -----------------------------------------------------
        # Semantic
        # -----------------------------------------------------

        semantic_results, _ = (
            sem_mod.search_with_temporal_filter(
                semantic_idx,
                sections,
                rec["query"],
                incident_date=incident_date,
                top_k=RETRIEVAL_K,
            )
        )

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        fused = fuse_results(
            bm25_results,
            semantic_results,
            alpha=ALPHA,
            top_k=RETRIEVAL_K,
        )

        fused_ranked = rank_keys(fused)

        fusion_rank = rank_of(
            fused_ranked,
            expected
        )

        # -----------------------------------------------------
        # Reranker
        # -----------------------------------------------------

        reranked = rerank_sections(
            query=rec["query"],
            hybrid_results=fused,
            top_k=TOP_K,
        )

        final_results = [
            item.hybrid_result
            for item in reranked
        ]

        reranked_ranked = rank_keys(
            final_results
        )

        rerank_rank = rank_of(
            reranked_ranked,
            expected
        )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        fusion_h1 = (
            fusion_rank == 1
        )

        fusion_h3 = (
            fusion_rank is not None
            and fusion_rank <= 3
        )

        rerank_h1 = (
            rerank_rank == 1
        )

        rerank_h3 = (
            rerank_rank is not None
            and rerank_rank <= 3
        )

        fusion_hit1 += int(fusion_h1)
        fusion_hit3 += int(fusion_h3)

        rerank_hit1 += int(rerank_h1)
        rerank_hit3 += int(rerank_h3)

        print()
        print("-" * 90)
        print(f"{rec['id']} — {rec['query']}")
        print(
            f"Expected: "
            f"{expected[0]} {expected[1]}"
        )

        print(
            f"Fusion:   "
            f"rank={fusion_rank} "
            f"top1={fused_ranked[0] if fused_ranked else None}"
        )

        print(
            f"Reranker: "
            f"rank={rerank_rank} "
            f"top1={reranked_ranked[0] if reranked_ranked else None}"
        )

        if fusion_rank != rerank_rank:
            print(">>> RANK CHANGED BY RERANKER")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    n = len(eval_records)

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print()
    print("WITHOUT RERANKER")
    print(
        f"Hit@1: "
        f"{fusion_hit1 / n * 100:.1f}%"
    )
    print(
        f"Hit@3: "
        f"{fusion_hit3 / n * 100:.1f}%"
    )

    print()
    print("WITH RERANKER")
    print(
        f"Hit@1: "
        f"{rerank_hit1 / n * 100:.1f}%"
    )
    print(
        f"Hit@3: "
        f"{rerank_hit3 / n * 100:.1f}%"
    )

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    if rerank_hit1 > fusion_hit1:
        print(
            "Reranker improves Top-1 accuracy."
        )
    elif rerank_hit1 < fusion_hit1:
        print(
            "WARNING: Reranker reduces Top-1 accuracy."
        )
    else:
        print(
            "Reranker does not change Top-1 accuracy."
        )

    if rerank_hit3 > fusion_hit3:
        print(
            "Reranker improves Top-3 accuracy."
        )
    elif rerank_hit3 < fusion_hit3:
        print(
            "WARNING: Reranker reduces Top-3 accuracy."
        )
    else:
        print(
            "Reranker does not change Top-3 accuracy."
        )


if __name__ == "__main__":
    main()