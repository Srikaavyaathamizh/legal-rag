"""
Compare Phase 9 System 4 vs System 5

System 4:
    Full system = temporal + BM25 + semantic + hybrid

System 5:
    Full system + section-aware reranker

This script identifies:
    - records fixed by reranker
    - records made worse by reranker
    - records unchanged
    - Hit@1 changes
    - Hit@3 changes
    - MRR changes
"""

from __future__ import annotations

import csv
import os
from collections import Counter


RESULTS_CSV = os.path.join(
    os.path.dirname(__file__),
    "ablation_results.csv"
)

OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "reranker_comparison.csv"
)


SYSTEM4 = "4_full_system"
SYSTEM5 = "5_full_system_reranked"


def load_results():
    rows = []

    with open(
        RESULTS_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            if row["system"] in {SYSTEM4, SYSTEM5}:
                rows.append(row)

    return rows


def parse_top_k(value):
    """
    Convert:

        IPC:304A | IPC:279 | IPC:269

    into:

        [('IPC', '304A'), ...]
    """

    if not value.strip():
        return []

    result = []

    for item in value.split(" | "):

        law, section = item.split(":", 1)

        result.append(
            (law, section)
        )

    return result


def compare():

    rows = load_results()

    by_id = {}

    for row in rows:

        eval_id = row["eval_id"]

        if eval_id not in by_id:
            by_id[eval_id] = {}

        by_id[eval_id][row["system"]] = row

    fixed = []
    worsened = []
    unchanged = []

    comparison_rows = []

    for eval_id, systems in by_id.items():

        if SYSTEM4 not in systems:
            continue

        if SYSTEM5 not in systems:
            continue

        s4 = systems[SYSTEM4]
        s5 = systems[SYSTEM5]

        expected = (
            s4["expected_law"],
            s4["expected_section"]
        )

        top4 = parse_top_k(
            s4["top_k_returned"]
        )

        top5 = parse_top_k(
            s5["top_k_returned"]
        )

        hit1_4 = int(s4["hit1"])
        hit1_5 = int(s5["hit1"])

        hit3_4 = int(s4["hit3"])
        hit3_5 = int(s5["hit3"])

        rr4 = float(s4["rr"])
        rr5 = float(s5["rr"])

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        if hit1_4 == 0 and hit1_5 == 1:
            status = "FIXED"

            fixed.append(eval_id)

        elif hit1_4 == 1 and hit1_5 == 0:
            status = "WORSENED"

            worsened.append(eval_id)

        else:
            status = "UNCHANGED"

            unchanged.append(eval_id)

        # --------------------------------------------------
        # Rank positions
        # --------------------------------------------------

        try:
            rank4 = top4.index(expected) + 1
        except ValueError:
            rank4 = None

        try:
            rank5 = top5.index(expected) + 1
        except ValueError:
            rank5 = None

        comparison_rows.append({
            "eval_id": eval_id,
            "category": s4["category"],
            "query": s4["query"],
            "expected_law": expected[0],
            "expected_section": expected[1],

            "system4_top1":
                f"{s4['top1_law']}:{s4['top1_section']}",

            "system5_top1":
                f"{s5['top1_law']}:{s5['top1_section']}",

            "system4_rank": rank4 if rank4 else "",
            "system5_rank": rank5 if rank5 else "",

            "system4_hit1": hit1_4,
            "system5_hit1": hit1_5,

            "system4_hit3": hit3_4,
            "system5_hit3": hit3_5,

            "system4_mrr": rr4,
            "system5_mrr": rr5,

            "mrr_change":
                round(rr5 - rr4, 4),

            "status": status
        })

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print("=" * 80)
    print("SYSTEM 4 vs SYSTEM 5 — RERANKER COMPARISON")
    print("=" * 80)

    print()

    print(f"Total records compared: {len(comparison_rows)}")

    print()

    print(f"FIXED by reranker:     {len(fixed)}")
    print(f"WORSENED by reranker:  {len(worsened)}")
    print(f"UNCHANGED:             {len(unchanged)}")

    print()
    print("=" * 80)
    print("FIXED CASES")
    print("=" * 80)

    for row in comparison_rows:

        if row["status"] != "FIXED":
            continue

        print()
        print(f"ID:       {row['eval_id']}")
        print(f"Category: {row['category']}")
        print(f"Query:    {row['query']}")
        print(
            f"Expected: "
            f"{row['expected_law']}:{row['expected_section']}"
        )
        print(
            f"System 4: {row['system4_top1']} "
            f"(rank={row['system4_rank']})"
        )
        print(
            f"System 5: {row['system5_top1']} "
            f"(rank={row['system5_rank']})"
        )

    print()
    print("=" * 80)
    print("WORSENED CASES")
    print("=" * 80)

    for row in comparison_rows:

        if row["status"] != "WORSENED":
            continue

        print()
        print(f"ID:       {row['eval_id']}")
        print(f"Category: {row['category']}")
        print(f"Query:    {row['query']}")
        print(
            f"Expected: "
            f"{row['expected_law']}:{row['expected_section']}"
        )
        print(
            f"System 4: {row['system4_top1']} "
            f"(rank={row['system4_rank']})"
        )
        print(
            f"System 5: {row['system5_top1']} "
            f"(rank={row['system5_rank']})"
        )

    # ------------------------------------------------------
    # Category summary
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("CATEGORY SUMMARY")
    print("=" * 80)

    categories = sorted(
        set(row["category"] for row in comparison_rows)
    )

    for category in categories:

        category_rows = [
            r for r in comparison_rows
            if r["category"] == category
        ]

        fixed_count = sum(
            r["status"] == "FIXED"
            for r in category_rows
        )

        worse_count = sum(
            r["status"] == "WORSENED"
            for r in category_rows
        )

        unchanged_count = sum(
            r["status"] == "UNCHANGED"
            for r in category_rows
        )

        mrr4 = sum(
            r["system4_mrr"]
            for r in category_rows
        ) / len(category_rows)

        mrr5 = sum(
            r["system5_mrr"]
            for r in category_rows
        ) / len(category_rows)

        print()
        print(category)
        print("-" * 40)
        print(f"Records:    {len(category_rows)}")
        print(f"Fixed:      {fixed_count}")
        print(f"Worsened:   {worse_count}")
        print(f"Unchanged:  {unchanged_count}")
        print(f"MRR System4: {mrr4:.3f}")
        print(f"MRR System5: {mrr5:.3f}")
        print(f"MRR change:  {mrr5 - mrr4:+.3f}")

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    fieldnames = list(
        comparison_rows[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(comparison_rows)

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print(
        f"Saved detailed comparison -> {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    compare()