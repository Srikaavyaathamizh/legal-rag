"""
Phase 2 — Temporal Filtering Logic
-----------------------------------
Determines which law (IPC or BNS) applies to a given incident date,
and filters the unified section corpus down to only the candidate
sections that a retriever (BM25 / semantic / hybrid) should search over.

This is a single-cutover system: BNS came into force 2024-07-01 and IPC
was repealed the same day. There are no per-section amendment dates to
track — the corpus's effective_from/effective_until fields already encode
this cutover, so this module's job is just to (a) resolve a law from a
date, and (b) apply that resolution safely, including edge cases.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Literal

CUTOVER_DATE = date(2024, 7, 1)  # BNS effective_from / day after IPC effective_until

LawCode = Literal["IPC", "BNS", "AMBIGUOUS"]


@dataclass
class LawResolution:
    law: LawCode
    reason: str
    confidence: Literal["high", "low"]
    warnings: list[str] = field(default_factory=list)


def resolve_applicable_law(incident_date: Optional[date]) -> LawResolution:
    """
    Core temporal rule. Given an incident date, decide which law applies.

    Edge cases handled explicitly:
      1. No date provided at all.
      2. Date exactly on the cutover boundary (2024-07-01).
      3. Dates far in the future (should still resolve to BNS, but flagged
         since the corpus can't know about laws not yet enacted).
    """
    if incident_date is None:
        return LawResolution(
            law="AMBIGUOUS",
            reason="No incident date was provided or extracted from the question.",
            confidence="low",
            warnings=[
                "Cannot determine IPC vs BNS without a date. "
                "Ask the user for the incident date, or show both law "
                "versions clearly labelled as unresolved."
            ],
        )

    if incident_date < CUTOVER_DATE:
        return LawResolution(
            law="IPC",
            reason=f"Incident date {incident_date} is before the BNS cutover "
                   f"({CUTOVER_DATE}), so the Indian Penal Code, 1860 applies.",
            confidence="high",
        )

    # incident_date >= CUTOVER_DATE
    warnings = []
    if incident_date > date.today():
        warnings.append(
            "Incident date is in the future relative to today; resolving to "
            "BNS on the assumption no further statutory replacement has "
            "occurred, but this cannot be verified from a static corpus."
        )

    return LawResolution(
        law="BNS",
        reason=f"Incident date {incident_date} is on or after the BNS cutover "
               f"({CUTOVER_DATE}), so the Bharatiya Nyaya Sanhita, 2023 applies.",
        confidence="high",
        warnings=warnings,
    )


def get_candidate_sections(
    all_sections: list[dict],
    incident_date: Optional[date],
) -> tuple[list[dict], LawResolution]:
    """
    Filters the full unified corpus down to the candidate set a retriever
    should search over, given an incident date.

    Returns (candidate_sections, resolution) so the caller / explainability
    layer always has access to *why* this filtering happened.
    """
    resolution = resolve_applicable_law(incident_date)

    if resolution.law == "AMBIGUOUS":
        # Safe default: don't guess. Return everything, tagged so the
        # UI/LLM layer knows it must ask for clarification rather than
        # silently answer from one law.
        return list(all_sections), resolution

    candidates = [
        s for s in all_sections
        if s["law"] == resolution.law
        and s["status"] in ("in_force", "repealed")  # excludes repealed_in_ipc, see below
    ]

    # Edge case: sections repealed *within* IPC before BNS ever existed
    # (status == "repealed_in_ipc"). We deliberately exclude these from
    # normal candidate search because they have empty text and no
    # reliable internal-repeal date in this dataset. If the resolved law
    # is IPC, we still want to know if the user's query concerns one of
    # these, so surface a warning rather than silently dropping context.
    dead_in_ipc = [s for s in all_sections if s["status"] == "repealed_in_ipc"]
    if resolution.law == "IPC" and dead_in_ipc:
        resolution.warnings.append(
            f"{len(dead_in_ipc)} IPC sections ({', '.join(s['section'] for s in dead_in_ipc)}) "
            "were already repealed within IPC itself before the BNS transition, and this "
            "dataset does not record their internal repeal dates. If the query concerns one "
            "of these sections, the IPC-applies conclusion may still be wrong for dates after "
            "their (unknown) internal repeal — flag for manual legal verification."
        )

    return candidates, resolution


def get_current_equivalent(section_record: dict, all_sections_by_key: dict) -> dict:
    """
    Given an IPC section record, look up its current (BNS) equivalent(s)
    using mapped_sections. Returns a structured result distinguishing
    three real cases in this corpus:
      - one clean mapping
      - many-to-one consolidation (multiple IPC sections -> one BNS section)
      - no equivalent at all (mapped_sections == ["REPEALED"])
    """
    mapped = section_record.get("mapped_sections", [])

    if mapped == ["REPEALED"]:
        return {
            "status": "no_current_equivalent",
            "message": f"IPC Section {section_record['section']} has no equivalent "
                       f"provision in BNS; it was repealed without replacement.",
        }

    equivalents = []
    for bns_sec in mapped:
        key = ("BNS", bns_sec)
        if key in all_sections_by_key:
            equivalents.append(all_sections_by_key[key])

    return {
        "status": "mapped",
        "count": len(equivalents),
        "consolidated": len(equivalents) > 1,  # informational; multi-target from one IPC section
        "equivalents": equivalents,
    }


def build_lookup_index(all_sections: list[dict]) -> dict:
    """Helper: (law, section) -> record, for O(1) mapped_sections lookups."""
    return {(s["law"], s["section"]): s for s in all_sections}
