from __future__ import annotations

import os
import sys
import json
from datetime import date as _date

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

FAILURES = {
    "EV1043": ("BNS", "127"),
    "EV1068": ("BNS", "179"),
    "EV1091": ("BNS", "309"),
}

RETRIEVAL_K = 30
ALPHA = 0.3


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


def find_section(sections, law, section):
    for record in sections:
        if (
            record.get("law") == law
            and record.get("section") == section
        ):
            return record
    return None


def find_result(results, expected):
    for rank, result in enumerate(results, start=1):
        if key(result.section_record) == expected:
            return rank, result

    return None, None


def show_section(record):
    if record is None:
        print("NOT FOUND")
        return

    print(f"Law:     {record.get('law')}")
    print(f"Section: {record.get('section')}")
    print(f"Status:  {record.get('status')}")
    print(f"Heading: {record.get('heading')}")
    print()
    print("TEXT:")
    print(record.get("text", ""))
    print()


def show_retrieval_result(label, rank, result):
    print(f"{label}")

    if result is None:
        print("  NOT RETRIEVED")
        return

    r = result.section_record

    print(f"  Rank:    {rank}")
    print(f"  Law:     {r.get('law')}")
    print(f"  Section: {r.get('section')}")
    print(f"  Heading: {r.get('heading')}")

    if hasattr(result, "score"):
        print(f"  Score:   {result.score:.6f}")

    print()


def main():

    sections, eval_records = load_data()

    eval_by_id = {
        rec["id"]: rec
        for rec in eval_records
    }

    print("=" * 100)
    print("REMAINING CONSOLIDATED FAILURE DIAGNOSIS")
    print("=" * 100)

    print(f"Alpha:       {ALPHA}")
    print(f"Retrieval K: {RETRIEVAL_K}")
    print()

    # ------------------------------------------------------------
    # Load indexes
    # ------------------------------------------------------------

    print("=" * 100)
    print("Loading BM25")
    print("=" * 100)

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    print("BM25 loaded.")
    print()

    print("=" * 100)
    print("Loading REAL SentenceTransformer + Qdrant")
    print("=" * 100)

    semantic_idx = sem_mod.SemanticIndex()
    semantic_idx.build(sections)

    print("Semantic backend:", type(semantic_idx).__name__)
    print()

    # ------------------------------------------------------------
    # Analyze each failure
    # ------------------------------------------------------------

    for eval_id, expected in FAILURES.items():

        rec = eval_by_id[eval_id]

        query = rec["query"]
        incident_date = parse_date(rec["incident_date"])

        print()
        print("=" * 100)
        print(f"{eval_id}")
        print("=" * 100)

        print(f"Query:         {query}")
        print(f"Incident date: {rec['incident_date']}")
        print(f"Expected:      {expected[0]} {expected[1]}")
        print()

        # --------------------------------------------------------
        # Expected section
        # --------------------------------------------------------

        expected_record = find_section(
            sections,
            expected[0],
            expected[1],
        )

        print("=" * 100)
        print("EXPECTED SECTION")
        print("=" * 100)

        show_section(expected_record)

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
        # Expected section rank
        # --------------------------------------------------------

        bm25_rank, bm25_expected = find_result(
            bm25_results,
            expected,
        )

        semantic_rank, semantic_expected = find_result(
            semantic_results,
            expected,
        )

        print("=" * 100)
        print("EXPECTED SECTION RETRIEVAL")
        print("=" * 100)

        show_retrieval_result(
            "BM25:",
            bm25_rank,
            bm25_expected,
        )

        show_retrieval_result(
            "Semantic:",
            semantic_rank,
            semantic_expected,
        )

        # --------------------------------------------------------
        # Fusion
        # --------------------------------------------------------

        fused = fuse_results(
            bm25_results,
            semantic_results,
            alpha=ALPHA,
            top_k=RETRIEVAL_K,
        )

        fused_rank = None
        fused_expected = None

        for rank, result in enumerate(fused, start=1):
            if key(result.section_record) == expected:
                fused_rank = rank
                fused_expected = result
                break

        print("=" * 100)
        print("FUSION")
        print("=" * 100)

        if fused_expected is None:
            print("Expected section NOT in fused results.")
        else:
            print(f"Expected fusion rank: {fused_rank}")
            print(f"BM25 raw:            {fused_expected.bm25_score_raw:.6f}")
            print(f"BM25 normalized:     {fused_expected.bm25_score_norm:.6f}")
            print(f"Semantic raw:        {fused_expected.semantic_score_raw:.6f}")
            print(f"Semantic normalized: {fused_expected.semantic_score_norm:.6f}")
            print(f"Fused score:         {fused_expected.fused_score:.6f}")

        print()

        # --------------------------------------------------------
        # Top fusion candidates
        # --------------------------------------------------------

        print("=" * 100)
        print("TOP 10 FUSION RESULTS")
        print("=" * 100)

        print(
            f"{'Rank':<6}"
            f"{'Law':<8}"
            f"{'Section':<15}"
            f"{'BM25':<12}"
            f"{'Semantic':<12}"
            f"{'Fused':<12}"
        )

        for rank, result in enumerate(fused[:10], start=1):

            r = result.section_record

            marker = ""

            if key(r) == expected:
                marker = "  <-- EXPECTED"

            print(
                f"{rank:<6}"
                f"{r.get('law', ''):<8}"
                f"{r.get('section', ''):<15}"
                f"{result.bm25_score_norm:<12.4f}"
                f"{result.semantic_score_norm:<12.4f}"
                f"{result.fused_score:<12.4f}"
                f"{marker}"
            )

        print()

        # --------------------------------------------------------
        # Diagnosis
        # --------------------------------------------------------

        print("=" * 100)
        print("INITIAL DIAGNOSIS")
        print("=" * 100)

        if semantic_rank is None and bm25_rank is None:
            print("RETRIEVAL FAILURE")
            print("Expected section is absent from both retrievers.")

        elif semantic_rank is not None and bm25_rank is None:
            print("SEMANTIC-ONLY RETRIEVAL")
            print(
                f"Expected section is retrieved semantically at rank "
                f"{semantic_rank}, but BM25 does not retrieve it."
            )

            if fused_rank is not None and fused_rank > 1:
                print(
                    "Fusion/ranking is likely responsible for the "
                    "remaining error."
                )

        elif bm25_rank is not None and semantic_rank is not None:
            print("BOTH RETRIEVERS FOUND EXPECTED")
            print(
                f"BM25 rank={bm25_rank}, "
                f"Semantic rank={semantic_rank}, "
                f"Fusion rank={fused_rank}"
            )

            if fused_rank is not None and fused_rank > min(
                bm25_rank,
                semantic_rank,
            ):
                print(
                    "FUSION HURTS — expected section is promoted "
                    "downward during fusion."
                )

        print()

        # --------------------------------------------------------
        # Inspect competitors
        # --------------------------------------------------------

        print("=" * 100)
        print("TOP COMPETITOR DETAILS")
        print("=" * 100)

        competitor_keys = [
            key(r.section_record)
            for r in fused[:5]
            if key(r.section_record) != expected
        ]

        for competitor_key in competitor_keys:

            competitor = find_section(
                sections,
                competitor_key[0],
                competitor_key[1],
            )

            print(
                f"\n--- {competitor_key[0]} {competitor_key[1]} ---"
            )

            if competitor:
                print(
                    f"Heading: {competitor.get('heading')}"
                )

                print(
                    f"Text: {competitor.get('text', '')[:1200]}"
                )

        print()


if __name__ == "__main__":
    main()