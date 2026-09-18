"""
Phase 8b — Automated Cross-Check of eval_dataset.json
----------------------------------------------------------
IMPORTANT SCOPE NOTE, read before trusting this output:

This script does NOT constitute legal verification. It cross-checks
each record in eval_dataset.json against data/raw/IPC_BNS_mapping.csv
directly, as a SECOND, INDEPENDENT re-derivation of the answer —
independent in the sense that it doesn't touch unified_sections.json
at all, so it catches bugs introduced anywhere in the Phase 1
(unification) or Phase 8 (eval generation) pipeline. It does NOT check
whether IPC_BNS_mapping.csv itself is correct against the actual
Gazette-published BNS concordance tables — that step still needs a
human (ideally with legal domain knowledge) to spot-check the CSV
against the primary source. Passing this script's checks is necessary
but not sufficient for the "hand-verified gold set" claim in your
paper; treat a PASS here as "internally consistent", not "legally
correct".

What it checks, per record:
  - IPC_STANDARD / NO_EQUIVALENT: does IPC_BNS_mapping.csv's row for
    this IPC section actually show the expected outcome (a specific
    BNS section for IPC_STANDARD, or the literal "Repealed in BNS"
    marker for NO_EQUIVALENT)?
  - BNS_STANDARD: does at least one row in the mapping CSV list this
    exact BNS_Section? (BNS sections are the "target" column, so this
    just confirms the section identifier genuinely exists in the raw
    source, not fabricated.)
  - CONSOLIDATED: does the mapping CSV's row for the stated IPC
    predecessor point to the expected consolidated BNS section?
  - Data-quality flags (separate from correctness): heading/query text
    containing a trailing stray digit (footnote-marker artifact seen
    earlier in the source CSVs), or unusually short/empty query text.

Records that fail ANY check are left verified=false with a note
explaining why, for a human to look at directly. Records that pass
are set to a provisional "auto_check": "PASS" field -- the human still
needs to flip "verified": true themselves; this script does not do
that automatically, to keep a real human decision in the loop for the
claim that actually goes in the paper.

Usage:
    python verify_eval_set.py
"""

from __future__ import annotations
import csv
import json
import os
import re

MAPPING_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "IPC_BNS_mapping.csv")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")  # updated in place

REPEALED_MARKER = "Repealed in BNS"  # canonical form used in messages; matching is case-insensitive below
FOOTNOTE_DIGIT_RE = re.compile(r"[a-z]\d+\??$")  # e.g. "...age2" or "...age2?"


def is_repealed_marker(value: str) -> bool:
    """
    The raw CSV itself has inconsistent capitalization for this marker
    ("Repealed in BNS" vs "Repealed In BNS" both occur) -- a genuine
    data-quality quirk in the source file, not something to silently
    paper over without noting it. Match case-insensitively rather than
    "fixing" the source file, so this script stays a read-only checker.
    """
    return value.strip().lower() == "repealed in bns"


def load_mapping():
    with open(MAPPING_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_ipc: dict[str, list[dict]] = {}
    by_bns: dict[str, list[dict]] = {}
    for r in rows:
        by_ipc.setdefault(r["IPC_Section"], []).append(r)
        bns_val = r["BNS_Section"]
        if not is_repealed_marker(bns_val):
            by_bns.setdefault(bns_val, []).append(r)
    return by_ipc, by_bns


def check_record(rec: dict, by_ipc: dict, by_bns: dict) -> tuple[bool, str]:
    cat = rec["category"]

    if cat == "IPC_STANDARD":
        rows = by_ipc.get(rec["source_section"])
        if not rows:
            return False, f"IPC section {rec['source_section']} not found in raw mapping CSV at all."
        bns_targets = {r["BNS_Section"] for r in rows}
        if any(is_repealed_marker(v) for v in bns_targets):
            return False, (
                f"Raw mapping CSV marks IPC {rec['source_section']} as repealed with no BNS "
                f"target, but this record was categorized IPC_STANDARD (expects a live mapping). "
                f"Category/pipeline mismatch -- needs investigation, not just a data typo."
            )
        return True, f"Raw mapping confirms IPC {rec['source_section']} -> BNS {sorted(bns_targets)}"

    if cat == "NO_EQUIVALENT":
        rows = by_ipc.get(rec["source_section"])
        if not rows:
            return False, f"IPC section {rec['source_section']} not found in raw mapping CSV at all."
        bns_targets = {r["BNS_Section"] for r in rows}
        if not all(is_repealed_marker(v) for v in bns_targets):
            return False, (
                f"Expected raw mapping to show the repealed marker for IPC {rec['source_section']}, "
                f"but found {sorted(bns_targets)} instead -- this section may NOT actually be a "
                f"true no-equivalent case; re-check."
            )
        return True, f"Raw mapping confirms IPC {rec['source_section']} is marked repealed with no BNS target"

    if cat == "BNS_STANDARD":
        if rec["expected_section"] not in by_bns:
            return False, f"BNS section {rec['expected_section']} does not appear as a target in the raw mapping CSV."
        return True, f"BNS {rec['expected_section']} confirmed present as a mapping target in raw CSV"

    if cat == "CONSOLIDATED":
        rows = by_ipc.get(rec["source_section"])
        if not rows:
            return False, f"Predecessor IPC section {rec['source_section']} not found in raw mapping CSV at all."
        bns_targets = {r["BNS_Section"] for r in rows}
        if rec["expected_section"] not in bns_targets:
            return False, (
                f"Raw mapping shows IPC {rec['source_section']} -> {sorted(bns_targets)}, "
                f"which does NOT include the expected consolidated target BNS {rec['expected_section']}."
            )
        return True, (
            f"Raw mapping confirms predecessor IPC {rec['source_section']} -> "
            f"BNS {rec['expected_section']} (consolidated)"
        )

    return False, f"Unknown category {cat!r}, cannot check."


def check_text_quality(rec: dict) -> list[str]:
    flags = []
    q = rec["query"]
    heading = rec["source_heading"]
    if len(q.strip()) < 15:
        flags.append("query unusually short, may not be a well-formed question")
    if FOOTNOTE_DIGIT_RE.search(heading):
        flags.append(
            f"source_heading ends in a bare digit ('{heading[-6:]}') -- likely a footnote-marker "
            f"artifact carried over from the raw CSV text, not a real part of the heading. "
            f"Worth stripping before this goes in a paper/demo."
        )
    if '"' in heading and heading.count('"') % 2 != 0:
        flags.append("unbalanced quote mark in heading -- possible extraction artifact")
    return flags


def main():
    by_ipc, by_bns = load_mapping()
    with open(EVAL_PATH) as f:
        records = json.load(f)

    n_pass = n_fail = n_textflag = 0
    for rec in records:
        ok, msg = check_record(rec, by_ipc, by_bns)
        text_flags = check_text_quality(rec)

        if ok:
            n_pass += 1
            rec["auto_check"] = "PASS"
            rec["auto_check_detail"] = msg
        else:
            n_fail += 1
            rec["auto_check"] = "FAIL"
            rec["auto_check_detail"] = msg
            rec["verifier_notes"] = (rec.get("verifier_notes") or "") + f" [AUTO-CHECK FAILED: {msg}]"

        if text_flags:
            n_textflag += 1
            rec["text_quality_flags"] = text_flags
        else:
            rec["text_quality_flags"] = []

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Checked {len(records)} records against data/raw/IPC_BNS_mapping.csv (independent source).")
    print(f"  auto_check PASS: {n_pass}")
    print(f"  auto_check FAIL: {n_fail}")
    print(f"  text quality flags raised: {n_textflag}")
    print(f"\nWrote annotated file back to {OUT_PATH}")
    print(
        "\nReminder: PASS here means 'consistent with the raw mapping CSV', not "
        "'legally correct'. A human still needs to review at least the FAIL rows "
        "(drop or fix them) and ideally spot-check a sample of PASS rows against "
        "the actual Gazette-published BNS concordance before setting verified=true "
        "and using this as the reported eval set."
    )

    if n_fail:
        print("\n--- FAILED RECORDS ---")
        for rec in records:
            if rec["auto_check"] == "FAIL":
                print(f"  {rec['id']} [{rec['category']}] {rec['source_law']} {rec['source_section']}: {rec['auto_check_detail']}")


if __name__ == "__main__":
    main()
