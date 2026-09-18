"""
Phase 9d — Alpha Sweep
---------------------------
Sweeps the BM25-vs-semantic fusion weight (alpha in hybrid_ranker.fuse_results,
final = alpha*BM25_norm + (1-alpha)*Semantic_norm) across the full [0,1]
range, on the temporal-filtered candidate set (i.e. this always uses the
correct law's candidates -- it isolates the fusion weight question from
the temporal-filtering question, which is already answered separately
in run_ablation.py).

Reports OVERALL and per-category Hit@1/Hit@3/MRR at each alpha, so you
can see directly whether alpha=0.5 (the default used everywhere else)
is actually optimal, or whether some categories (e.g. CONSOLIDATED)
would do better semantic-heavy (low alpha) or BM25-heavy (high alpha).

Retrieval itself (BM25 top-30, semantic top-30) only needs to happen
ONCE per eval record -- fusion at different alphas is then just cheap
re-weighting of the same two pre-retrieved lists (see hybrid_ranker.
sweep_alpha, which this script calls directly rather than re-querying
per alpha).

Usage:
    python alpha_sweep.py
"""

from __future__ import annotations
import sys, os, json
from datetime import date as _date
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bm25_index as bm25_mod
from hybrid_ranker import sweep_alpha

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
OUT_CSV = os.path.join(os.path.dirname(__file__), "alpha_sweep_results.csv")

ALPHAS = [round(x * 0.1, 1) for x in range(11)]  # 0.0, 0.1, ..., 1.0
RETRIEVAL_K = 30
TOP_K = 10


def parse_date(s: str) -> _date:
    y, m, d = (int(x) for x in s.split("-"))
    return _date(y, m, d)


def score(ranked_keys, expected):
    hit1 = 1 if ranked_keys and ranked_keys[0] == expected else 0
    hit3 = 1 if expected in ranked_keys[:3] else 0
    rr = 0.0
    for i, k in enumerate(ranked_keys, start=1):
        if k == expected:
            rr = 1.0 / i
            break
    return hit1, hit3, rr


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        sections = json.load(f)["sections"]
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_records = json.load(f)

    bm25_idx = bm25_mod.LawScopedBM25(sections)

    import semantic_index as sem_mod

    print("=" * 80)
    print("Loading REAL SentenceTransformer + Qdrant semantic backend")
    print("=" * 80)

    semantic_idx = sem_mod.SemanticIndex()
    semantic_idx.build(sections)

    print("Semantic backend:", type(semantic_idx).__name__)

    print(f"Loaded {len(sections)} sections, {len(eval_records)} eval records.")
    print(f"Sweeping alpha over {ALPHAS}\n")

    # raw[alpha][category] -> list of (hit1, hit3, rr)
    raw = defaultdict(lambda: defaultdict(list))

    for rec in eval_records:
        expected = (rec["expected_law"], rec["expected_section"])
        incident_date = parse_date(rec["incident_date"])

        bm25_results, _ = bm25_mod.search_with_temporal_filter(
            bm25_idx, sections, rec["query"], incident_date=incident_date, top_k=RETRIEVAL_K
        )
        semantic_results, _ = sem_mod.search_with_temporal_filter(
            semantic_idx, sections, rec["query"], incident_date=incident_date, top_k=RETRIEVAL_K
        )

        # one call, reused across all alphas -- see module docstring
        sweep = sweep_alpha(bm25_results, semantic_results, alphas=ALPHAS, top_k=TOP_K)

        for alpha, fused in sweep.items():
            ranked_keys = [(r.section_record["law"], r.section_record["section"]) for r in fused]
            hit1, hit3, rr = score(ranked_keys, expected)
            raw[alpha]["OVERALL"].append((hit1, hit3, rr))
            raw[alpha][rec["category"]].append((hit1, hit3, rr))

    def agg(triples):
        n = len(triples)
        return {
            "n": n,
            "hit1": sum(t[0] for t in triples) / n,
            "hit3": sum(t[1] for t in triples) / n,
            "mrr": sum(t[2] for t in triples) / n,
        }

    categories = ["OVERALL", "IPC_STANDARD", "BNS_STANDARD", "NO_EQUIVALENT", "CONSOLIDATED"]

    csv_rows = []
    for cat in categories:
        print("=" * 90)
        print(f"CATEGORY: {cat}")
        print(f"{'alpha':>6s} {'n':>4s} {'Hit@1':>8s} {'Hit@3':>8s} {'MRR':>8s}")
        for alpha in ALPHAS:
            if cat not in raw[alpha]:
                continue
            m = agg(raw[alpha][cat])
            marker = "  <- current default" if alpha == 0.5 else ""
            print(f"{alpha:>6.1f} {m['n']:>4d} {m['hit1']*100:>7.1f}% {m['hit3']*100:>7.1f}% {m['mrr']:>8.3f}{marker}")
            csv_rows.append({"category": cat, "alpha": alpha, **m})
        # flag the best alpha for this category by Hit@1, ties broken by MRR
        best = max(ALPHAS, key=lambda a: (agg(raw[a][cat])["hit1"], agg(raw[a][cat])["mrr"]))
        best_m = agg(raw[best][cat])
        print(f"  best alpha for {cat} by Hit@1: {best} (Hit@1={best_m['hit1']*100:.1f}%, MRR={best_m['mrr']:.3f})")
        print()

    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "alpha", "n", "hit1", "hit3", "mrr"])
        writer.writeheader()
        for row in csv_rows:
            row = dict(row)
            row["hit1"] = round(row["hit1"], 4)
            row["hit3"] = round(row["hit3"], 4)
            row["mrr"] = round(row["mrr"], 4)
            writer.writerow(row)
    print(f"Wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
