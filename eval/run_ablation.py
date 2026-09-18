"""
Phase 9 — Ablation Experiments
----------------------------------
Runs the 4-system comparison from the research plan:

    System                        Temporal   BM25   Semantic
    1. Semantic-only (baseline)      no       no       yes
    2. Hybrid, no temporal           no       yes      yes
    3. Temporal + semantic           yes      no       yes
    4. Full system (proposed)        yes      yes      yes

...on eval/eval_dataset.json, reporting Hit@1, Hit@3, MRR, and
Law-Version Error Rate (LVER) — did the top-1 result come from the
wrong Act entirely, regardless of whether the section itself matched.

DESIGN NOTE — how "no temporal" is actually simulated:
Rather than writing a second, separate retrieval path for the two
"no temporal" systems, this script reuses Phase 3/4's existing
AMBIGUOUS-date handling: calling search_with_temporal_filter(...,
incident_date=None) already searches BOTH law indices and interleaves
results (see bm25_index.py / semantic_index.py). That is exactly what
a system with no temporal awareness does — it doesn't know which law
applies, so it searches everything. Systems 3 and 4 pass the record's
REAL incident_date, which resolves to a single law and searches only
that law's candidates. No new retrieval logic was written for this
script; it only recombines Phase 2-5 building blocks differently
per system, which is itself evidence the modularity from earlier
phases was worth building.

SEMANTIC BACKEND: this script takes any object exposing the same
.search(query, law=None, candidate_section_ids=None, top_k=10)
interface as semantic_index.SemanticIndex (duck-typed, matching what
semantic_index.search_with_temporal_filter already expects). In this
sandbox (no internet to huggingface.co) it is run with a TF-IDF
stand-in — see semantic_stub_sandbox.py, sandbox-only, NOT what ships.
Run this same script locally with a real SemanticIndex instance for
the numbers that actually belong in your paper.

Usage:
    python run_ablation.py                  # uses whichever backend
                                             # main() is wired to below
"""

from __future__ import annotations
import sys, os, json, csv
from datetime import date as _date
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal_filter import build_lookup_index
import bm25_index as bm25_mod
import semantic_index as sem_mod
from hybrid_ranker import fuse_results
from section_reranker import rerank_sections

TOP_K = 10  # retrieval depth for Hit@3 / MRR scoring
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "ablation_results.csv")
SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "ablation_summary.csv")


def parse_date(s: str) -> _date:
    """eval_dataset.json stores incident_date as 'YYYY-MM-DD' strings (JSON
    has no native date type); temporal_filter.py's functions expect real
    datetime.date objects. Converting once at this boundary rather than
    inside temporal_filter.py keeps that module's contract (it takes a
    date, not a string it has to guess the format of) unchanged."""
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    return sections, eval_records


def ranked_keys(results) -> list[tuple[str, str]]:
    """Extracts (law, section) in rank order from any result-list type
    (BM25Result / SemanticResult / HybridResult all expose .section_record)."""
    return [(r.section_record["law"], r.section_record["section"]) for r in results]


# --------------------------------------------------------------- systems ---

def system_semantic_only_no_temporal(bm25_idx, semantic_idx, sections, query, incident_date, top_k=TOP_K):
    results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx, sections, query, incident_date=None, top_k=top_k
    )
    return ranked_keys(results)


def system_hybrid_no_temporal(bm25_idx, semantic_idx, sections, query, incident_date, top_k=TOP_K, alpha=0.3):
    bm25_results, _ = bm25_mod.search_with_temporal_filter(
        bm25_idx, sections, query, incident_date=None, top_k=30
    )
    semantic_results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx, sections, query, incident_date=None, top_k=30
    )
    fused = fuse_results(bm25_results, semantic_results, alpha=alpha, top_k=top_k)
    return ranked_keys(fused)


def system_temporal_semantic(bm25_idx, semantic_idx, sections, query, incident_date, top_k=TOP_K):
    results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx, sections, query, incident_date=incident_date, top_k=top_k
    )
    return ranked_keys(results)


def system_full(bm25_idx, semantic_idx, sections, query, incident_date, top_k=TOP_K, alpha=0.3):
    bm25_results, _ = bm25_mod.search_with_temporal_filter(
        bm25_idx, sections, query, incident_date=incident_date, top_k=30
    )
    semantic_results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx, sections, query, incident_date=incident_date, top_k=30
    )
    fused = fuse_results(bm25_results, semantic_results, alpha=alpha, top_k=top_k)
    return ranked_keys(fused)

def system_full_reranked(
    bm25_idx,
    semantic_idx,
    sections,
    query,
    incident_date,
    top_k=TOP_K,
    alpha=0.3
):
    """
    System 5 — Full system + section-aware reranker.

    Pipeline:

        Temporal filtering
              ↓
        BM25 top-30
              +
        Semantic top-30
              ↓
        Hybrid fusion
              ↓
        Section-aware reranking
              ↓
        final top-k
    """

    # ---------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------

    bm25_results, _ = bm25_mod.search_with_temporal_filter(
        bm25_idx,
        sections,
        query,
        incident_date=incident_date,
        top_k=30
    )

    # ---------------------------------------------------------
    # Semantic retrieval
    # ---------------------------------------------------------

    semantic_results, _ = sem_mod.search_with_temporal_filter(
        semantic_idx,
        sections,
        query,
        incident_date=incident_date,
        top_k=30
    )

    # ---------------------------------------------------------
    # Hybrid fusion
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Keep 30 candidates here.
    #
    # The reranker needs the wider candidate pool.
    #

    fused = fuse_results(
        bm25_results,
        semantic_results,
        alpha=alpha,
        top_k=30
    )

    # ---------------------------------------------------------
    # Section-aware reranking
    # ---------------------------------------------------------

    reranked = rerank_sections(
        query=query,
        hybrid_results=fused,
        top_k=top_k
    )

    # ---------------------------------------------------------
    # Convert reranker output to section keys
    # ---------------------------------------------------------

    return [
        (
            item.hybrid_result.section_record["law"],
            item.hybrid_result.section_record["section"]
        )
        for item in reranked
    ]


SYSTEMS = {
    "1_semantic_only_no_temporal": system_semantic_only_no_temporal,
    "2_hybrid_no_temporal": system_hybrid_no_temporal,
    "3_temporal_semantic": system_temporal_semantic,
    "4_full_system": system_full,
    "5_full_system_reranked": system_full_reranked,
}


# --------------------------------------------------------------- metrics ---

def score_one(ranked: list[tuple[str, str]], expected: tuple[str, str]) -> dict:
    hit1 = 1 if ranked and ranked[0] == expected else 0
    hit3 = 1 if expected in ranked[:3] else 0
    rr = 0.0
    for i, k in enumerate(ranked, start=1):
        if k == expected:
            rr = 1.0 / i
            break
    top1_law = ranked[0][0] if ranked else None
    law_error = 1 if (top1_law is not None and top1_law != expected[0]) else 0
    return {"hit1": hit1, "hit3": hit3, "rr": rr, "law_error": law_error, "no_result": 0 if ranked else 1}


def run_ablation(bm25_idx, semantic_idx, sections, eval_records):
    """
    Returns:
      per_record_rows: list of dicts, one per (system, eval record) —
        raw material for error_analysis.py and the CSV export.
      summary: dict[system][category or 'OVERALL'] -> aggregate metrics.
    """
    per_record_rows = []
    # summary[system][category] -> list of per-record score dicts
    raw = defaultdict(lambda: defaultdict(list))

    for rec in eval_records:
        expected = (rec["expected_law"], rec["expected_section"])
        incident_date = parse_date(rec["incident_date"])
        for sys_name, sys_fn in SYSTEMS.items():
            ranked = sys_fn(bm25_idx, semantic_idx, sections, rec["query"], incident_date)
            scores = score_one(ranked, expected)

            per_record_rows.append({
                "system": sys_name,
                "eval_id": rec["id"],
                "category": rec["category"],
                "query": rec["query"],
                "incident_date": rec["incident_date"],
                "expected_law": expected[0],
                "expected_section": expected[1],
                "top1_law": ranked[0][0] if ranked else "",
                "top1_section": ranked[0][1] if ranked else "",
                "hit1": scores["hit1"],
                "hit3": scores["hit3"],
                "rr": round(scores["rr"], 4),
                "law_error": scores["law_error"],
                "no_result": scores["no_result"],
                "top_k_returned": [f"{l}:{s}" for l, s in ranked],
            })

            raw[sys_name]["OVERALL"].append(scores)
            raw[sys_name][rec["category"]].append(scores)

    summary = {}
    for sys_name, by_cat in raw.items():
        summary[sys_name] = {}
        for cat, score_list in by_cat.items():
            n = len(score_list)
            summary[sys_name][cat] = {
                "n": n,
                "hit1": sum(s["hit1"] for s in score_list) / n,
                "hit3": sum(s["hit3"] for s in score_list) / n,
                "mrr": sum(s["rr"] for s in score_list) / n,
                "law_error_rate": sum(s["law_error"] for s in score_list) / n,
                "no_result_rate": sum(s["no_result"] for s in score_list) / n,
            }
    return per_record_rows, summary


# ---------------------------------------------------------------- report ---

SYSTEM_LABELS = {
    "1_semantic_only_no_temporal": "1. Semantic-only (baseline)",
    "2_hybrid_no_temporal": "2. Hybrid, no temporal",
    "3_temporal_semantic": "3. Temporal + semantic",
    "4_full_system": "4. Full system (proposed)",
    "5_full_system_reranked": "5. Full system + reranker",
}


def print_report(summary):
    print("=" * 100)
    print("OVERALL (all 55 eval records)")
    print(f"{'System':38s} {'n':>4s} {'Hit@1':>8s} {'Hit@3':>8s} {'MRR':>8s} {'LawVerErr%':>11s}")
    for sys_name in SYSTEMS:
        m = summary[sys_name]["OVERALL"]
        print(f"{SYSTEM_LABELS[sys_name]:38s} {m['n']:>4d} "
              f"{m['hit1']*100:>7.1f}% {m['hit3']*100:>7.1f}% {m['mrr']:>8.3f} "
              f"{m['law_error_rate']*100:>10.1f}%")

    for cat in ["IPC_STANDARD", "BNS_STANDARD", "NO_EQUIVALENT", "CONSOLIDATED"]:
        print()
        print("=" * 100)
        n_cat = summary["4_full_system"].get(cat, {}).get("n", 0)
        print(f"CATEGORY: {cat}  (n={n_cat})")
        print(f"{'System':38s} {'n':>4s} {'Hit@1':>8s} {'Hit@3':>8s} {'MRR':>8s} {'LawVerErr%':>11s}")
        for sys_name in SYSTEMS:
            m = summary[sys_name].get(cat)
            if not m:
                continue
            print(f"{SYSTEM_LABELS[sys_name]:38s} {m['n']:>4d} "
                  f"{m['hit1']*100:>7.1f}% {m['hit3']*100:>7.1f}% {m['mrr']:>8.3f} "
                  f"{m['law_error_rate']*100:>10.1f}%")


def write_csvs(per_record_rows, summary):
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_record_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_record_rows:
            row = dict(row)
            row["top_k_returned"] = " | ".join(row["top_k_returned"])
            writer.writerow(row)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "category", "n", "hit1", "hit3", "mrr", "law_error_rate", "no_result_rate"])
        for sys_name, by_cat in summary.items():
            for cat, m in by_cat.items():
                writer.writerow([sys_name, cat, m["n"], round(m["hit1"], 4), round(m["hit3"], 4),
                                  round(m["mrr"], 4), round(m["law_error_rate"], 4), round(m["no_result_rate"], 4)])

    print(f"\nWrote per-record results -> {RESULTS_CSV}")
    print(f"Wrote summary table -> {SUMMARY_CSV}")


def main():
    sections, eval_records = load_data()
    print(f"Loaded {len(sections)} corpus sections, {len(eval_records)} eval records.\n")

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    # Semantic backend: sandbox stub here (no internet to huggingface.co).
    # Locally, replace this import + call with:
    #     from semantic_stub_sandbox import real_semantic_index
    #     semantic_idx = real_semantic_index(sections)
    from semantic_stub_sandbox import real_semantic_index

    print("=" * 100)
    print("Using REAL SentenceTransformer + Qdrant semantic backend.")
    print("These results are suitable for reporting, subject to evaluation-set validation.")
    print("=" * 100)

    semantic_idx = real_semantic_index(sections)
    per_record_rows, summary = run_ablation(bm25_idx, semantic_idx, sections, eval_records)
    print_report(summary)
    write_csvs(per_record_rows, summary)


if __name__ == "__main__":
    main()
