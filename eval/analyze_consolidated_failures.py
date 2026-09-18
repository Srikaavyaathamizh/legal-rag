"""
Phase 9c — Automated CONSOLIDATED Failure Classifier
----------------------------------------------------------
Turns the manual "paste error_analysis_report.md back and forth" loop
into one script. For every CONSOLIDATED eval record, runs semantic-only,
BM25-only, and full-hybrid retrieval separately, finds the RANK of the
expected (law, section) answer in each list independently, and
classifies the failure pattern:

  FUSION_HURTS       — semantic (or BM25) alone ranks the expected
                        answer #1, but the fused/hybrid system does not.
                        Actionable via alpha, not a new model.
  GENUINE_MISS       — expected answer doesn't appear in EITHER
                        individual retriever's top-K at all. A real
                        retrieval/representation gap, not a fusion or
                        reranking problem.
  RANKING_ONLY       — expected answer IS in the hybrid system's top-3,
                        just not top-1. Recall is fine; this is exactly
                        what a reranker is *supposed* to fix.
  CORRECT            — hybrid top-1 already matches (not a failure at
                        all; included for completeness/contrast).

This only classifies CONSOLIDATED records, since that's where the
open question is -- run it on the whole eval set by removing the
category filter if you want the same breakdown elsewhere.

IMPORTANT: run this with your REAL semantic backend (SentenceTransformer
+ Qdrant), not the TF-IDF sandbox stand-in, before trusting the
classification for your paper -- see semantic_stub_sandbox.py's
docstring for why TF-IDF numbers aren't comparable to your real run.

Usage:
    python analyze_consolidated_failures.py
"""

from __future__ import annotations
import sys, os, json
from datetime import date as _date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bm25_index as bm25_mod
from hybrid_ranker import fuse_results

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
OUT_MD = os.path.join(os.path.dirname(__file__), "consolidated_failure_analysis.md")

RETRIEVAL_K = 30  # how deep each individual retriever searches, before fusion


def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def rank_of(ranked_keys: list[tuple[str, str]], expected: tuple[str, str]) -> int | None:
    """1-indexed rank of expected in a ranked (law, section) list, or None if absent."""
    for i, k in enumerate(ranked_keys, start=1):
        if k == expected:
            return i
    return None


def classify(sem_rank, bm25_rank, hybrid_rank) -> str:
    if hybrid_rank == 1:
        return "CORRECT"
    # Did either individual retriever alone nail #1, while fusion didn't?
    if (sem_rank == 1 or bm25_rank == 1) and hybrid_rank != 1:
        return "FUSION_HURTS"
    # Is it missing from BOTH individual lists entirely?
    if sem_rank is None and bm25_rank is None:
        return "GENUINE_MISS"
    # Otherwise: present somewhere, just not ranked #1 by either alone
    # or by fusion -- check if hybrid at least got it into top-3.
    if hybrid_rank is not None and hybrid_rank <= 3:
        return "RANKING_ONLY"
    return "GENUINE_MISS"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    consolidated = [r for r in eval_records if r["category"] == "CONSOLIDATED"]
    print(f"Analyzing {len(consolidated)} CONSOLIDATED records...")

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    # Swap this import for your real semantic_index.SemanticIndex locally.
    from semantic_stub_sandbox import real_semantic_index
    semantic_idx = real_semantic_index(sections)

    import semantic_index as sem_mod

    rows = []
    for rec in consolidated:
        expected = (rec["expected_law"], rec["expected_section"])
        incident_date = parse_date(rec["incident_date"])

        bm25_results, _ = bm25_mod.search_with_temporal_filter(
            bm25_idx, sections, rec["query"], incident_date=incident_date, top_k=RETRIEVAL_K
        )
        semantic_results, _ = sem_mod.search_with_temporal_filter(
            semantic_idx, sections, rec["query"], incident_date=incident_date, top_k=RETRIEVAL_K
        )
        fused = fuse_results(bm25_results, semantic_results, alpha=0.5, top_k=RETRIEVAL_K)

        bm25_keys = [(r.section_record["law"], r.section_record["section"]) for r in bm25_results]
        sem_keys = [(r.section_record["law"], r.section_record["section"]) for r in semantic_results]
        hybrid_keys = [(r.section_record["law"], r.section_record["section"]) for r in fused]

        sem_rank = rank_of(sem_keys, expected)
        bm25_rank = rank_of(bm25_keys, expected)
        hybrid_rank = rank_of(hybrid_keys, expected)

        diagnosis = classify(sem_rank, bm25_rank, hybrid_rank)

        rows.append({
            "id": rec["id"],
            "query": rec["query"],
            "expected": f"{expected[0]} {expected[1]}",
            "semantic_rank": sem_rank,
            "bm25_rank": bm25_rank,
            "hybrid_rank": hybrid_rank,
            "hybrid_top1": f"{hybrid_keys[0][0]} {hybrid_keys[0][1]}" if hybrid_keys else "(none)",
            "diagnosis": diagnosis,
        })

    from collections import Counter
    diag_counts = Counter(r["diagnosis"] for r in rows)

    print("\n" + "=" * 100)
    print(f"{'ID':8s} {'Expected':14s} {'Sem rank':>9s} {'BM25 rank':>10s} {'Hybrid rank':>12s} {'Diagnosis':16s}")
    for r in rows:
        print(f"{r['id']:8s} {r['expected']:14s} "
              f"{str(r['semantic_rank']):>9s} {str(r['bm25_rank']):>10s} {str(r['hybrid_rank']):>12s} "
              f"{r['diagnosis']:16s}")

    print("\nDiagnosis breakdown:")
    for diag, n in diag_counts.items():
        print(f"  {diag:16s} {n}")

    # -------- write markdown report --------
    lines = [
        "# Phase 9c -- CONSOLIDATED Failure Classification\n",
        "**Scope note:** each record's expected (law, section) answer is looked up "
        "independently in the semantic-only ranked list, the BM25-only ranked list, "
        "and the fused hybrid list (alpha=0.5). Classification logic:\n",
        "- `FUSION_HURTS`: one individual retriever alone ranked the expected answer #1, "
        "but fusion did not -- actionable via alpha, not a new model.\n",
        "- `GENUINE_MISS`: expected answer absent from BOTH individual retrievers' top-"
        f"{RETRIEVAL_K} entirely -- a real representation/retrieval gap.\n",
        "- `RANKING_ONLY`: expected answer is in the hybrid system's top-3, just not "
        "top-1 -- recall is fine, this is what a reranker should fix.\n",
        "- `CORRECT`: hybrid top-1 already matches (not a failure).\n",
        "\n## Per-record results\n",
        "| ID | Expected | Semantic rank | BM25 rank | Hybrid rank | Hybrid top-1 | Diagnosis |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['expected']} | {r['semantic_rank']} | {r['bm25_rank']} | "
            f"{r['hybrid_rank']} | {r['hybrid_top1']} | {r['diagnosis']} |"
        )
    lines.append("\n## Diagnosis breakdown\n")
    for diag, n in diag_counts.items():
        lines.append(f"- **{diag}**: {n}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote -> {OUT_MD}")


if __name__ == "__main__":
    main()
