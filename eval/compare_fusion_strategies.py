"""
Phase 9g — Fusion Strategy Comparison

Compares three fusion strategies on the FULL evaluation set:

1. Current weighted fusion, alpha=0.5
2. Best validated weighted fusion, alpha=0.3
3. Reciprocal Rank Fusion (RRF), k=60

Uses the REAL SentenceTransformer + Qdrant backend.

Purpose:
Determine whether the remaining fusion errors are better handled by
rank-based fusion rather than min-max score fusion.

This is an evaluation experiment only.
It does NOT modify hybrid_ranker.py.
"""

from __future__ import annotations

import sys
import os
import json
import csv
from datetime import date as _date
from collections import defaultdict

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

import bm25_index as bm25_mod
import semantic_index as sem_mod
from hybrid_ranker import fuse_results


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

OUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "fusion_strategy_comparison.csv",
)

RETRIEVAL_K = 30
TOP_K = 10
RRF_K = 60

ALPHAS = [0.5, 0.3]


def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def result_key(result):
    return (
        result.section_record["law"],
        result.section_record["section"],
    )


def score(ranked_keys, expected):
    hit1 = (
        1
        if ranked_keys and ranked_keys[0] == expected
        else 0
    )

    hit3 = (
        1
        if expected in ranked_keys[:3]
        else 0
    )

    rr = 0.0

    for i, key in enumerate(ranked_keys, start=1):
        if key == expected:
            rr = 1.0 / i
            break

    return hit1, hit3, rr


def rrf_fuse(
    bm25_results,
    semantic_results,
    k=60,
    top_k=10,
):
    """
    Reciprocal Rank Fusion.

    RRF score:

        sum(1 / (k + rank))

    Rank is 1-based.

    Unlike min-max score fusion, RRF does not compare BM25
    score magnitudes with semantic similarity magnitudes.
    """

    scores = {}
    records = {}

    # BM25 contribution
    for rank, result in enumerate(
        bm25_results,
        start=1
    ):
        key = result_key(result)

        scores.setdefault(key, 0.0)
        scores[key] += 1.0 / (k + rank)

        records[key] = result.section_record

    # Semantic contribution
    for rank, result in enumerate(
        semantic_results,
        start=1
    ):
        key = result_key(result)

        scores.setdefault(key, 0.0)
        scores[key] += 1.0 / (k + rank)

        records[key] = result.section_record

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            item[1],
            item[0][0],
            item[0][1],
        ),
        reverse=True,
    )

    return [
        (
            key,
            records[key],
            score,
        )
        for key, score in ranked[:top_k]
    ]


def ranked_keys_from_hybrid(results):
    return [
        result_key(result)
        for result in results
    ]


def ranked_keys_from_rrf(results):
    return [
        item[0]
        for item in results
    ]


def main():

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    with open(
        DATA_PATH,
        encoding="utf-8"
    ) as f:
        sections = json.load(f)["sections"]

    with open(
        EVAL_PATH,
        encoding="utf-8"
    ) as f:
        eval_records = json.load(f)

    print("=" * 90)
    print("PHASE 9g — FUSION STRATEGY COMPARISON")
    print("=" * 90)

    print(
        f"Corpus sections: {len(sections)}"
    )

    print(
        f"Evaluation records: {len(eval_records)}"
    )

    print(
        f"Retrieval K: {RETRIEVAL_K}"
    )

    print(
        f"Final K: {TOP_K}"
    )

    print(
        f"RRF k: {RRF_K}"
    )

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("Loading BM25 backend")
    print("=" * 90)

    bm25_idx = bm25_mod.LawScopedBM25(
        sections
    )

    print(
        "BM25 backend:",
        type(bm25_idx).__name__
    )

    # ---------------------------------------------------------
    # REAL semantic backend
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("Loading REAL SentenceTransformer + Qdrant backend")
    print("=" * 90)

    semantic_idx = sem_mod.SemanticIndex()
    semantic_idx.build(sections)

    print(
        "Semantic backend:",
        type(semantic_idx).__name__
    )

    # ---------------------------------------------------------
    # Strategy names
    # ---------------------------------------------------------

    strategies = [
        "alpha_0.5",
        "alpha_0.3",
        "rrf_k60",
    ]

    raw = defaultdict(
        lambda: defaultdict(list)
    )

    csv_rows = []

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    for i, rec in enumerate(
        eval_records,
        start=1
    ):

        expected = (
            rec["expected_law"],
            rec["expected_section"],
        )

        incident_date = parse_date(
            rec["incident_date"]
        )

        # -----------------------------------------------------
        # Retrieve ONCE
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
        # Alpha 0.5
        # -----------------------------------------------------

        fused_05 = fuse_results(
            bm25_results,
            semantic_results,
            alpha=0.5,
            top_k=TOP_K,
        )

        ranked_05 = ranked_keys_from_hybrid(
            fused_05
        )

        # -----------------------------------------------------
        # Alpha 0.3
        # -----------------------------------------------------

        fused_03 = fuse_results(
            bm25_results,
            semantic_results,
            alpha=0.3,
            top_k=TOP_K,
        )

        ranked_03 = ranked_keys_from_hybrid(
            fused_03
        )

        # -----------------------------------------------------
        # RRF
        # -----------------------------------------------------

        rrf_results = rrf_fuse(
            bm25_results,
            semantic_results,
            k=RRF_K,
            top_k=TOP_K,
        )

        ranked_rrf = ranked_keys_from_rrf(
            rrf_results
        )

        rankings = {
            "alpha_0.5": ranked_05,
            "alpha_0.3": ranked_03,
            "rrf_k60": ranked_rrf,
        }

        # -----------------------------------------------------
        # Score every strategy
        # -----------------------------------------------------

        for strategy, ranked in rankings.items():

            hit1, hit3, rr = score(
                ranked,
                expected
            )

            raw[strategy]["OVERALL"].append(
                (hit1, hit3, rr)
            )

            raw[strategy][
                rec["category"]
            ].append(
                (hit1, hit3, rr)
            )

            csv_rows.append({
                "strategy": strategy,
                "eval_id": rec["id"],
                "category": rec["category"],
                "query": rec["query"],
                "expected_law": expected[0],
                "expected_section": expected[1],
                "top1_law": (
                    ranked[0][0]
                    if ranked
                    else ""
                ),
                "top1_section": (
                    ranked[0][1]
                    if ranked
                    else ""
                ),
                "hit1": hit1,
                "hit3": hit3,
                "rr": round(rr, 4),
                "ranking": " | ".join(
                    f"{law}:{section}"
                    for law, section in ranked
                ),
            })

        if i % 10 == 0:
            print(
                f"Processed {i}/{len(eval_records)}"
            )

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    def aggregate(values):

        n = len(values)

        return {
            "n": n,
            "hit1": (
                sum(x[0] for x in values) / n
            ),
            "hit3": (
                sum(x[1] for x in values) / n
            ),
            "mrr": (
                sum(x[2] for x in values) / n
            ),
        }

    categories = [
        "OVERALL",
        "IPC_STANDARD",
        "BNS_STANDARD",
        "NO_EQUIVALENT",
        "CONSOLIDATED",
    ]

    # ---------------------------------------------------------
    # Print report
    # ---------------------------------------------------------

    for category in categories:

        print()
        print("=" * 90)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 90)

        print(
            f"{'Strategy':20s}"
            f"{'n':>5s}"
            f"{'Hit@1':>10s}"
            f"{'Hit@3':>10s}"
            f"{'MRR':>10s}"
        )

        for strategy in strategies:

            values = raw[strategy].get(
                category,
                []
            )

            if not values:
                continue

            m = aggregate(values)

            print(
                f"{strategy:20s}"
                f"{m['n']:>5d}"
                f"{m['hit1'] * 100:>9.1f}%"
                f"{m['hit3'] * 100:>9.1f}%"
                f"{m['mrr']:>10.3f}"
            )

    # ---------------------------------------------------------
    # Find best overall
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("BEST STRATEGY")
    print("=" * 90)

    best_strategy = max(
        strategies,
        key=lambda s: (
            aggregate(
                raw[s]["OVERALL"]
            )["hit1"],
            aggregate(
                raw[s]["OVERALL"]
            )["mrr"],
        ),
    )

    best_metrics = aggregate(
        raw[best_strategy]["OVERALL"]
    )

    print(
        f"Best overall: {best_strategy}"
    )

    print(
        f"Hit@1: {best_metrics['hit1'] * 100:.1f}%"
    )

    print(
        f"Hit@3: {best_metrics['hit3'] * 100:.1f}%"
    )

    print(
        f"MRR: {best_metrics['mrr']:.3f}"
    )

    # ---------------------------------------------------------
    # Compare consolidated failures
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("CONSOLIDATED RECORD COMPARISON")
    print("=" * 90)

    consolidated_records = [
        rec for rec in eval_records
        if rec["category"] == "CONSOLIDATED"
    ]

    for rec in consolidated_records:

        expected = (
            rec["expected_law"],
            rec["expected_section"],
        )

        rows_for_record = [
            row
            for row in csv_rows
            if row["eval_id"] == rec["id"]
        ]

        print()
        print(
            f"{rec['id']} — {rec['query']}"
        )

        print(
            f"Expected: "
            f"{expected[0]} {expected[1]}"
        )

        for row in rows_for_record:

            marker = (
                "✓"
                if row["hit1"] == 1
                else "✗"
            )

            print(
                f"  {row['strategy']:12s} "
                f"-> "
                f"{row['top1_law']} "
                f"{row['top1_section']} "
                f"{marker}"
            )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                csv_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print("=" * 90)
    print(
        f"Saved -> {OUT_CSV}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()