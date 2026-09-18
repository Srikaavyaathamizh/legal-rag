"""
Phase 9c — Candidate Depth / Recall Analysis

Purpose:
Determine whether the expected legal section is actually retrieved
by BM25 and Semantic retrieval before fusion.

This separates:

1. Retrieval failure
   -> expected section is absent from both candidate pools

2. Ranking failure
   -> expected section is present in candidates but ranked poorly

Runs on the REAL SentenceTransformer + Qdrant backend.
"""

from __future__ import annotations

import os
import sys
import json
import csv
from datetime import date as _date

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

import bm25_index as bm25_mod
import semantic_index as sem_mod


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

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "candidate_depth_results.csv",
)

DEPTHS = [5, 10, 20, 30, 50, 100]


def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def load_data():

    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    return sections, eval_records


def key(record):
    return (
        record["law"],
        record["section"],
    )


def result_key(result):
    return key(result.section_record)


def rank_of(results, expected):

    for i, result in enumerate(results, start=1):

        if result_key(result) == expected:
            return i

    return None


def main():

    sections, eval_records = load_data()

    print(
        f"Loaded {len(sections)} corpus sections, "
        f"{len(eval_records)} eval records."
    )

    # ============================================================
    # REAL BM25
    # ============================================================

    print("=" * 80)
    print("Loading BM25 backend")
    print("=" * 80)

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    print(
        "BM25 backend:",
        type(bm25_idx).__name__
    )

    # ============================================================
    # REAL SENTENCE TRANSFORMER + QDRANT
    # ============================================================

    print("=" * 80)
    print("Loading REAL SentenceTransformer + Qdrant semantic backend")
    print("=" * 80)

    semantic_idx = sem_mod.SemanticIndex()

    semantic_idx.build(sections)

    print(
        "Semantic backend:",
        type(semantic_idx).__name__
    )

    # ============================================================
    # RETRIEVAL
    # ============================================================

    rows = []

    max_depth = max(DEPTHS)

    for i, record in enumerate(eval_records, start=1):

        query = record["query"]

        incident_date = parse_date(
            record["incident_date"]
        )

        expected = (
            record["expected_law"],
            record["expected_section"],
        )

        category = record["category"]
        eval_id = record["id"]

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        bm25_results, _ = (
            bm25_mod.search_with_temporal_filter(
                bm25_idx,
                sections,
                query,
                incident_date=incident_date,
                top_k=max_depth,
            )
        )

        # --------------------------------------------------------
        # Semantic
        # --------------------------------------------------------

        semantic_results, _ = (
            sem_mod.search_with_temporal_filter(
                semantic_idx,
                sections,
                query,
                incident_date=incident_date,
                top_k=max_depth,
            )
        )

        # --------------------------------------------------------
        # Expected section rank
        # --------------------------------------------------------

        bm25_rank = rank_of(
            bm25_results,
            expected
        )

        semantic_rank = rank_of(
            semantic_results,
            expected
        )

        # --------------------------------------------------------
        # Check each depth
        # --------------------------------------------------------

        for depth in DEPTHS:

            bm25_found = (
                bm25_rank is not None
                and bm25_rank <= depth
            )

            semantic_found = (
                semantic_rank is not None
                and semantic_rank <= depth
            )

            union_found = (
                bm25_found
                or semantic_found
            )

            # Both retrievers independently found it
            both_found = (
                bm25_found
                and semantic_found
            )

            rows.append({

                "eval_id": eval_id,

                "category": category,

                "query": query,

                "expected_law": expected[0],

                "expected_section": expected[1],

                "depth": depth,

                "bm25_rank": (
                    bm25_rank
                    if bm25_rank is not None
                    else ""
                ),

                "semantic_rank": (
                    semantic_rank
                    if semantic_rank is not None
                    else ""
                ),

                "bm25_found": int(
                    bm25_found
                ),

                "semantic_found": int(
                    semantic_found
                ),

                "union_found": int(
                    union_found
                ),

                "both_found": int(
                    both_found
                ),
            })

        if i % 10 == 0:

            print(
                f"Processed "
                f"{i}/{len(eval_records)}"
            )

    # ============================================================
    # SAVE CSV
    # ============================================================

    fields = [
        "eval_id",
        "category",
        "query",
        "expected_law",
        "expected_section",
        "depth",
        "bm25_rank",
        "semantic_rank",
        "bm25_found",
        "semantic_found",
        "union_found",
        "both_found",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)

    # ============================================================
    # OVERALL RECALL
    # ============================================================

    print()
    print("=" * 80)
    print("OVERALL CANDIDATE RECALL")
    print("=" * 80)

    for depth in DEPTHS:

        subset = [
            r for r in rows
            if int(r["depth"]) == depth
        ]

        n = len(subset)

        bm25_recall = (
            sum(r["bm25_found"] for r in subset)
            / n
            * 100
        )

        semantic_recall = (
            sum(r["semantic_found"] for r in subset)
            / n
            * 100
        )

        union_recall = (
            sum(r["union_found"] for r in subset)
            / n
            * 100
        )

        both_recall = (
            sum(r["both_found"] for r in subset)
            / n
            * 100
        )

        print(
            f"K={depth:3d} | "
            f"BM25={bm25_recall:6.1f}% | "
            f"Semantic={semantic_recall:6.1f}% | "
            f"Union={union_recall:6.1f}% | "
            f"Both={both_recall:6.1f}%"
        )

    # ============================================================
    # CONSOLIDATED RECALL
    # ============================================================

    print()
    print("=" * 80)
    print("CONSOLIDATED CANDIDATE RECALL")
    print("=" * 80)

    consolidated_rows = [
        r for r in rows
        if r["category"] == "CONSOLIDATED"
    ]

    for depth in DEPTHS:

        subset = [
            r for r in consolidated_rows
            if int(r["depth"]) == depth
        ]

        n = len(subset)

        bm25_recall = (
            sum(r["bm25_found"] for r in subset)
            / n
            * 100
        )

        semantic_recall = (
            sum(r["semantic_found"] for r in subset)
            / n
            * 100
        )

        union_recall = (
            sum(r["union_found"] for r in subset)
            / n
            * 100
        )

        print(
            f"K={depth:3d} | "
            f"BM25={bm25_recall:6.1f}% | "
            f"Semantic={semantic_recall:6.1f}% | "
            f"Union={union_recall:6.1f}%"
        )

    # ============================================================
    # IMPORTANT: SHOW REMAINING FULL-SYSTEM FAILURES
    # ============================================================

    print()
    print("=" * 80)
    print("CONSOLIDATED FAILURE DIAGNOSIS")
    print("=" * 80)

    # IDs from your current System 4 failures
    failure_ids = {
        "EV1020",
        "EV1043",
        "EV1048",
        "EV1068",
        "EV1091",
    }

    for eval_id in failure_ids:

        record_rows = [
            r for r in rows
            if r["eval_id"] == eval_id
            and int(r["depth"]) == 30
        ]

        if not record_rows:
            continue

        r = record_rows[0]

        print()
        print(f"ID: {eval_id}")
        print(f"Category: {r['category']}")
        print(f"Expected: {r['expected_law']} {r['expected_section']}")

        print(
            f"BM25 rank:     "
            f"{r['bm25_rank'] or 'NOT FOUND'}"
        )

        print(
            f"Semantic rank: "
            f"{r['semantic_rank'] or 'NOT FOUND'}"
        )

        print(
            f"Union found:   "
            f"{'YES' if r['union_found'] else 'NO'}"
        )

        if r["union_found"]:

            print(
                "Diagnosis: "
                "RANKING / FUSION PROBLEM"
            )

        else:

            print(
                "Diagnosis: "
                "RETRIEVAL / CANDIDATE RECALL PROBLEM"
            )

    print()
    print("=" * 80)
    print(f"Saved results -> {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()