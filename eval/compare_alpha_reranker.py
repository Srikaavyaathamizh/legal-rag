"""
Phase 9e — Full System Alpha Validation

Compares the complete System 5 pipeline at:
    alpha = 0.5  (current)
    alpha = 0.4  (candidate)

Pipeline:
    Temporal filtering
        ↓
    BM25 + SentenceTransformer
        ↓
    Hybrid fusion
        ↓
    Section-aware reranker
        ↓
    Final ranking

Uses the REAL SentenceTransformer + Qdrant backend.
"""

from __future__ import annotations
import csv
import sys
import os
import json
from datetime import date as _date
from collections import defaultdict

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

import bm25_index as bm25_mod
import semantic_index as sem_mod

from hybrid_ranker import fuse_results
from section_reranker import rerank_sections


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

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

RESULTS_CSV = os.path.join(
    os.path.dirname(__file__),
    "alpha_reranker_results.csv",
)

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

ALPHAS = [0.3, 0.5, 0.4]

RETRIEVAL_K = 30
TOP_K = 10


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    return sections, eval_records


def rank_keys(results):
    return [
        (
            r.section_record["law"],
            r.section_record["section"]
        )
        for r in results
    ]


def score(ranked, expected):
    hit1 = (
        1
        if ranked and ranked[0] == expected
        else 0
    )

    hit3 = (
        1
        if expected in ranked[:3]
        else 0
    )

    rr = 0.0

    for i, item in enumerate(ranked, start=1):
        if item == expected:
            rr = 1.0 / i
            break

    law_error = (
        1
        if ranked and ranked[0][0] != expected[0]
        else 0
    )

    return {
        "hit1": hit1,
        "hit3": hit3,
        "rr": rr,
        "law_error": law_error,
    }


# ------------------------------------------------------------
# System 5
# ------------------------------------------------------------

def run_system_5(
    bm25_idx,
    semantic_idx,
    sections,
    query,
    incident_date,
    alpha,
):
    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results, _ = bm25_mod.search_with_temporal_filter(
        bm25_idx,
        sections,
        query,
        incident_date=incident_date,
        top_k=RETRIEVAL_K,
    )

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    semantic_results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx,
        sections,
        query,
        incident_date=incident_date,
        top_k=RETRIEVAL_K,
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    fused = fuse_results(
        bm25_results,
        semantic_results,
        alpha=alpha,
        top_k=RETRIEVAL_K,
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranked = rerank_sections(
        query=query,
        hybrid_results=fused,
        top_k=TOP_K,
    )

    return [
        (
            item.hybrid_result.section_record["law"],
            item.hybrid_result.section_record["section"],
        )
        for item in reranked
    ]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    sections, eval_records = load_data()

    print("=" * 90)
    print("PHASE 9e — FULL SYSTEM ALPHA VALIDATION")
    print("=" * 90)

    print(
        f"Corpus sections: {len(sections)}"
    )

    print(
        f"Evaluation records: {len(eval_records)}"
    )

    print(
        f"Testing alpha values: {ALPHAS}"
    )

    print()

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print("=" * 90)
    print("Loading BM25 backend")
    print("=" * 90)

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    print(
        "BM25 backend:",
        type(bm25_idx).__name__
    )

    print()

    # --------------------------------------------------------
    # Build REAL semantic backend
    # --------------------------------------------------------

    print("=" * 90)
    print("Loading REAL SentenceTransformer + Qdrant backend")
    print("=" * 90)

    semantic_idx = sem_mod.SemanticIndex()

    semantic_idx.build(sections)

    print(
        "Semantic backend:",
        type(semantic_idx).__name__
    )

    print()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    raw = defaultdict(
        lambda: defaultdict(list)
    )

    record_results = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for index, rec in enumerate(
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

        for alpha in ALPHAS:

            ranked = run_system_5(
                bm25_idx=bm25_idx,
                semantic_idx=semantic_idx,
                sections=sections,
                query=rec["query"],
                incident_date=incident_date,
                alpha=alpha,
            )

            scores = score(
                ranked,
                expected
            )

            category = rec["category"]

            raw[alpha]["OVERALL"].append(
                scores
            )

            raw[alpha][category].append(
                scores
            )

            record_results.append({
               "eval_id": rec["id"],
               "category": category,
               "query": rec["query"],
               "incident_date": rec["incident_date"],

               "alpha": alpha,

               "expected_law": expected[0],
               "expected_section": expected[1],

               "top1_law": ranked[0][0] if ranked else "",
               "top1_section": ranked[0][1] if ranked else "",

               "hit1": scores["hit1"],
               "hit3": scores["hit3"],
               "rr": scores["rr"],
               "law_error": scores["law_error"],

               "final_ranking": " | ".join(
               f"{law}:{section}"
               for law, section in ranked
                ),
                })

        if index % 10 == 0:
            print(
                f"Processed "
                f"{index}/{len(eval_records)}"
            )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    def aggregate(items):

        n = len(items)

        return {
            "n": n,
            "hit1": (
                sum(x["hit1"] for x in items)
                / n
            ),
            "hit3": (
                sum(x["hit3"] for x in items)
                / n
            ),
            "mrr": (
                sum(x["rr"] for x in items)
                / n
            ),
            "law_error": (
                sum(x["law_error"] for x in items)
                / n
            ),
        }

    categories = [
        "OVERALL",
        "IPC_STANDARD",
        "BNS_STANDARD",
        "NO_EQUIVALENT",
        "CONSOLIDATED",
    ]

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FULL SYSTEM + RERANKER COMPARISON")
    print("=" * 90)

    for category in categories:

        print()
        print("-" * 90)
        print(
            f"{category}"
        )
        print("-" * 90)

        print(
            f"{'Alpha':>8} "
            f"{'n':>5} "
            f"{'Hit@1':>10} "
            f"{'Hit@3':>10} "
            f"{'MRR':>10} "
            f"{'LawErr':>10}"
        )

        for alpha in ALPHAS:

            metrics = aggregate(
                raw[alpha][category]
            )

            marker = (
                "  CURRENT"
                if alpha == 0.5
                else "  CANDIDATE"
            )

            print(
                f"{alpha:>8.1f} "
                f"{metrics['n']:>5} "
                f"{metrics['hit1'] * 100:>9.1f}% "
                f"{metrics['hit3'] * 100:>9.1f}% "
                f"{metrics['mrr']:>10.3f} "
                f"{metrics['law_error'] * 100:>9.1f}%"
                f"{marker}"
            )

    # --------------------------------------------------------
    # Direct comparison
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("ALPHA 0.5 → ALPHA 0.4 CHANGE")
    print("=" * 90)

    for category in categories:

        old = aggregate(
            raw[0.5][category]
        )

        new = aggregate(
            raw[0.4][category]
        )

        hit1_delta = (
            new["hit1"] - old["hit1"]
        )

        hit3_delta = (
            new["hit3"] - old["hit3"]
        )

        mrr_delta = (
            new["mrr"] - old["mrr"]
        )

        print()
        print(category)

        print(
            f"  Hit@1: "
            f"{old['hit1'] * 100:.1f}% → "
            f"{new['hit1'] * 100:.1f}% "
            f"({hit1_delta * 100:+.1f} pp)"
        )

        print(
            f"  Hit@3: "
            f"{old['hit3'] * 100:.1f}% → "
            f"{new['hit3'] * 100:.1f}% "
            f"({hit3_delta * 100:+.1f} pp)"
        )

        print(
            f"  MRR: "
            f"{old['mrr']:.3f} → "
            f"{new['mrr']:.3f} "
            f"({mrr_delta:+.3f})"
        )

    # --------------------------------------------------------
    # Show records changed
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("RECORDS WHERE TOP-1 CHANGED")
    print("=" * 90)

    by_id = defaultdict(dict)

    for row in record_results:
        by_id[row["eval_id"]][row["alpha"]] = row

    changed = 0

    for eval_id, values in by_id.items():

        old = values[0.5]
        new = values[0.4]

        if old["top1_section"] != new["top1_section"]:

            changed += 1

            print()
            print(
                f"{eval_id} "
                f"[{old['category']}]"
            )

            print(
                f"  Expected: "
                f"{old['expected_law']} "
                f"{old['expected_section']}"
            )

            print(
                f"  α=0.5: "
                f"{old['top1_law']} "
                f"{old['top1_section']} "
                f"{'✓' if old['hit1'] else '✗'}"
            )

            print(
                f"  α=0.4: "
                f"{new['top1_law']} "
                f"{new['top1_section']} "
                f"{'✓' if new['hit1'] else '✗'}"
            )

    if changed == 0:
        print("No top-1 records changed.")

    print()
    print("=" * 90)
    print(
        f"Total top-1 changes: {changed}"
    )
    print("=" * 90)
    
    
        # --------------------------------------------------------
    # Save per-record results
    # --------------------------------------------------------

    fields = [
        "eval_id",
        "category",
        "query",
        "incident_date",
        "alpha",
        "expected_law",
        "expected_section",
        "top1_law",
        "top1_section",
        "hit1",
        "hit3",
        "rr",
        "law_error",
        "final_ranking",
    ]

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(record_results)

    print()
    print("=" * 90)
    print("PER-RECORD RESULTS")
    print("=" * 90)
    print(f"Saved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()