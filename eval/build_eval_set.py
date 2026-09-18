"""
Phase 8 — Evaluation Set Builder
-----------------------------------
Auto-generates a large pool of (question, incident_date, expected_law,
expected_section) test cases from the ALREADY-VERIFIED ground truth
that lives in unified_sections.json's `mapped_sections` field (itself
built from IPC_BNS_mapping.csv in Phase 1). We generate from
unified_sections.json rather than re-reading the raw mapping CSV
directly, because unified_sections.json is the exact artifact the
retrieval pipeline queries against — generating test cases from a
second, independently-parsed copy of the same mapping risks silently
testing against a different ground truth than what the system
actually sees.

Four question categories, matching the four things this project's
research claim needs to demonstrate:

  A. IPC_STANDARD      — ordinary IPC section, pre-cutover date.
                          expected_law=IPC, expected_section=<that section>.
  B. BNS_STANDARD       — ordinary BNS section, post-cutover date.
                          expected_law=BNS, expected_section=<that section>.
  C. NO_EQUIVALENT      — IPC section explicitly marked REPEALED with no
                          BNS successor (29 total, 25 with real text).
                          Stress-tests the "no current equivalent" path,
                          not just retrieval.
  D. CONSOLIDATED       — BNS sections that absorbed >1 IPC predecessor
                          (40 such BNS sections). One question is
                          generated PER predecessor, phrased using that
                          predecessor's own heading, with a POST-cutover
                          date. Expected_law=BNS, expected_section=the
                          single consolidated target — this is the
                          "can hybrid retrieval disambiguate near-duplicate
                          merged old sections" stress test.

Output:
  data/eval/eval_pool_full.csv       — everything generated (traceable,
                                        large; your paper trail)
  data/eval/eval_set_to_verify.csv   — a stratified subset (default 55)
                                        for a human to hand-check before
                                        it becomes the reported eval set.
                                        Comes with empty
                                        human_verified / verifier_notes /
                                        corrected_expected_section columns.

Usage:
    python build_eval_set.py
"""

from __future__ import annotations
import json
import csv
import random
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eval")

PRE_CUTOVER_DATE = "2023-06-01"   # comfortably inside the IPC era (cutover = 2024-07-01)
POST_CUTOVER_DATE = "2025-01-01"  # comfortably inside the BNS era

RANDOM_SEED = 42  # fixed, so the verify-subset sample is reproducible run to run

# How many rows to pull into the hand-verify subset, per category.
# Totals ~55; adjust to taste, but keep C and D generous relative to
# their small population sizes since they're the highest-value stress tests.
VERIFY_QUOTA = {
    "IPC_STANDARD": 18,
    "BNS_STANDARD": 15,
    "NO_EQUIVALENT": 12,   # out of only 25 available -- nearly half, deliberately
    "CONSOLIDATED": 10,
}


def load_sections() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["sections"]


def make_question(heading: str) -> str:
    """
    Deterministic, simple template on purpose (see module docstring —
    reproducibility over variety). heading text from the corpus is
    already a clean legal description, e.g. "Causing death by
    negligence" -> "What is the legal provision regarding causing
    death by negligence?"
    """
    h = heading.strip().strip(".")
    h = h[0].lower() + h[1:] if h else h
    return f"What is the legal provision regarding {h}?"


def detect_ambiguous_clusters(sections: list[dict]) -> dict[tuple[str, str], list[str]]:
    """
    Identifies sections that share an identical heading with at least one
    other section under the same law (e.g. BNS 318, 318(2), 318(3), 318(4)
    all have heading "Cheating"). make_question() only templates off the
    heading, so any two sections in the same cluster generate the EXACT
    SAME question text while expecting different "correct" answers --
    the eval question is unanswerable by any retriever, however good,
    because the query gives no information to distinguish them.

    Returns {(law, section): [all section ids sharing its heading]} for
    every section that is NOT uniquely identified by its own heading.
    This is used to route ambiguous sections into a separate,
    explicitly-flagged category rather than silently scoring them
    against the standard categories.
    """
    def normalize(h: str) -> str:
        return h.strip().lower().rstrip(".")

    clusters: dict[tuple[str, str], list[str]] = {}
    for s in sections:
        if not s["text"].strip():
            continue
        key = (s["law"], normalize(s["heading"]))
        clusters.setdefault(key, []).append(s["section"])

    ambiguous: dict[tuple[str, str], list[str]] = {}
    for (law, _), secs in clusters.items():
        if len(secs) > 1:
            for sec in secs:
                ambiguous[(law, sec)] = secs
    return ambiguous


def build_pool(sections: list[dict]) -> list[dict]:
    lookup = {(s["law"], s["section"]): s for s in sections}
    ambiguous_clusters = detect_ambiguous_clusters(sections)
    rows = []
    rid = 0

    def add(category, section_record, incident_date, expected_law, expected_section, note=""):
        nonlocal rid
        rid += 1
        # If this section's heading is shared by other sections under the
        # same law, the auto-generated question can't distinguish which
        # one is "correct" -- reroute to a clearly-flagged category
        # instead of silently scoring an unanswerable question against
        # IPC_STANDARD / BNS_STANDARD / CONSOLIDATED.
        key = (expected_law, expected_section)
        if key in ambiguous_clusters:
            category = f"{category}_AMBIGUOUS"
            note = (
                f"{note} [AUTO-FLAGGED: heading shared with sibling sections "
                f"{ambiguous_clusters[key]} -- question text cannot distinguish "
                f"which is correct without additional detail from the section body]"
            ).strip()
        rows.append({
            "id": f"EV{rid:04d}",
            "category": category,
            "query": make_question(section_record["heading"]),
            "incident_date": incident_date,
            "expected_law": expected_law,
            "expected_section": expected_section,
            "source_law": section_record["law"],
            "source_section": section_record["section"],
            "source_heading": section_record["heading"],
            "notes": note,
        })

    # --- Category A: IPC_STANDARD ---
    ipc_standard = [
        s for s in sections
        if s["law"] == "IPC" and s["status"] != "repealed_in_ipc"
        and s.get("mapped_sections") != ["REPEALED"]
        and s["text"].strip()
    ]
    for s in ipc_standard:
        add("IPC_STANDARD", s, PRE_CUTOVER_DATE, "IPC", s["section"])

    # --- Category B: BNS_STANDARD ---
    bns_standard = [s for s in sections if s["law"] == "BNS" and s["text"].strip()]
    for s in bns_standard:
        add("BNS_STANDARD", s, POST_CUTOVER_DATE, "BNS", s["section"])

    # --- Category C: NO_EQUIVALENT (repealed, no BNS successor) ---
    no_equivalent = [
        s for s in sections
        if s["law"] == "IPC" and s.get("mapped_sections") == ["REPEALED"] and s["text"].strip()
    ]
    for s in no_equivalent:
        add(
            "NO_EQUIVALENT", s, PRE_CUTOVER_DATE, "IPC", s["section"],
            note="expected current_equivalent.status == 'no_current_equivalent'",
        )

    # --- Category D: CONSOLIDATED (many IPC predecessors -> one BNS section) ---
    consolidated_bns = [s for s in sections if s["law"] == "BNS" and len(s.get("mapped_sections") or []) > 1]
    for bns_s in consolidated_bns:
        predecessor_ids = bns_s["mapped_sections"]
        for pred_id in predecessor_ids:
            pred = lookup.get(("IPC", pred_id))
            if pred is None or not pred["text"].strip():
                continue
            add(
                "CONSOLIDATED", pred, POST_CUTOVER_DATE, "BNS", bns_s["section"],
                note=(
                    f"phrased from IPC {pred['section']} heading, but post-cutover date "
                    f"-> query pool asks 'what applies now'; BNS {bns_s['section']} "
                    f"consolidates {len(predecessor_ids)} old sections: {predecessor_ids}"
                ),
            )

    return rows


def stratified_verify_sample(rows: list[dict], quota: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    sample = []
    for cat, n in quota.items():
        pool = by_cat.get(cat, [])
        n = min(n, len(pool))
        sample.extend(rng.sample(pool, n))
    # keep deterministic, readable ordering in the output file
    sample.sort(key=lambda r: (r["category"], r["id"]))
    return sample


def write_csv(path: str, rows: list[dict], extra_columns: list[str] | None = None):
    fieldnames = ["id", "category", "query", "incident_date", "expected_law", "expected_section",
                  "source_law", "source_section", "source_heading", "notes"]
    if extra_columns:
        fieldnames += extra_columns
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            for col in (extra_columns or []):
                row.setdefault(col, "")
            writer.writerow(row)


def main():
    sections = load_sections()
    pool = build_pool(sections)

    from collections import Counter
    counts = Counter(r["category"] for r in pool)
    print("Full auto-generated pool:")
    for cat, n in sorted(counts.items()):
        flag = "  <- excluded from graded categories, see below" if cat.endswith("_AMBIGUOUS") else ""
        print(f"  {cat:28s} {n}{flag}")
    print(f"  {'TOTAL':28s} {len(pool)}")

    ambiguous_total = sum(n for cat, n in counts.items() if cat.endswith("_AMBIGUOUS"))
    if ambiguous_total:
        print(
            f"\n{ambiguous_total} auto-generated questions were flagged _AMBIGUOUS: their section "
            f"shares an identical heading with at least one sibling section under the same law, "
            f"so the templated question cannot distinguish which one is 'correct'. These are "
            f"EXCLUDED from the graded IPC_STANDARD/BNS_STANDARD/NO_EQUIVALENT/CONSOLIDATED "
            f"categories and from the stratified hand-verify sample below, since scoring them "
            f"would penalize the retriever for an eval-set defect, not a retrieval failure. "
            f"They are still written to eval_pool_full.csv for transparency and future work "
            f"(e.g. hand-authoring distinguishing question text using the section body)."
        )

    full_path = os.path.join(OUT_DIR, "eval_pool_full.csv")
    write_csv(full_path, pool)
    print(f"\nWrote full pool -> {full_path}")

    verify_sample = stratified_verify_sample(pool, VERIFY_QUOTA, RANDOM_SEED)
    verify_counts = Counter(r["category"] for r in verify_sample)
    print("\nStratified hand-verify subset:")
    for cat, n in verify_counts.items():
        print(f"  {cat:15s} {n}")
    print(f"  TOTAL           {len(verify_sample)}")

    verify_path = os.path.join(OUT_DIR, "eval_set_to_verify.csv")
    write_csv(
        verify_path, verify_sample,
        extra_columns=["human_verified", "verifier_notes", "corrected_expected_section"],
    )
    print(f"\nWrote hand-verify subset -> {verify_path}")
    print(
        "\nNext step: open eval_set_to_verify.csv, and for each row confirm the "
        "query/date/expected_law/expected_section is actually correct against the "
        "primary source. Set human_verified=TRUE (or FALSE + a note in verifier_notes "
        "if something's wrong, and fill corrected_expected_section if you're fixing "
        "it rather than dropping it). Only TRUE rows should feed Phase 9 scoring."
    )


if __name__ == "__main__":
    main()
