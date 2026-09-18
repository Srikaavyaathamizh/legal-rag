"""
Phase 6 — Explainability Layer
---------------------------------
Turns a raw HybridResult (Phase 5) into the structured explanation the
UI/paper actually wants to show: not just "why is this score high" but
"why is this the applicable law at all", "what current/historical
provision does this correspond to", and "which words actually matched".

This module deliberately does NOT claim retrieval scores are legal proof
(see Session 10 in the plan: "these scores are retrieval evidence, not
legal certainty"). The `disclaimer` field on every Explanation exists so
that constraint is carried through the data, not just remembered by
whoever writes the UI later.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from temporal_filter import LawResolution, get_current_equivalent, build_lookup_index
from bm25_index import tokenize

DISCLAIMER = (
    "Retrieval scores indicate how strongly this provision matched the "
    "query and time period. They are retrieval evidence, not a legal "
    "determination — verify against the primary source before relying "
    "on this for any legal decision."
)

# Small stopword set used ONLY for the matched-keywords display (not for
# BM25 itself, which already handles term weighting via IDF). This is
# purely cosmetic: filters out words too common to be informative when
# shown to a user as "why this matched".
DISPLAY_STOPWORDS = {
    "the", "a", "an", "of", "or", "and", "any", "to", "by", "in", "for",
    "is", "was", "be", "shall", "with", "as", "that", "this", "on", "such",
}


@dataclass
class CurrentEquivalentInfo:
    direction: str  # "ipc_to_bns" | "bns_to_ipc"
    status: str     # "mapped" | "no_current_equivalent" | "no_predecessor"
    message: str
    sections: list[dict] = field(default_factory=list)  # full section records, if any


@dataclass
class Explanation:
    law: str
    section: str
    heading: str
    full_text: str
    law_reason: str
    law_confidence: str
    law_warnings: list[str]

    bm25_score_raw: float
    bm25_score_norm: float
    semantic_score_raw: float
    semantic_score_norm: float
    fused_score: float

    matched_keywords: list[str]

    current_equivalent: CurrentEquivalentInfo

    disclaimer: str = DISCLAIMER


def _matched_keywords(query: str, section_record: dict) -> list[str]:
    """
    Query tokens that also appear in the section's heading+text, in query
    order, deduplicated, with display stopwords removed. This is a
    transparency aid (show the user *why* BM25 fired), not a scoring
    mechanism — BM25's own IDF-weighted scoring already happened upstream.
    """
    query_tokens = tokenize(query)
    section_tokens = set(tokenize(f"{section_record['heading']} {section_record['text']}"))

    seen = set()
    matched = []
    for tok in query_tokens:
        if tok in DISPLAY_STOPWORDS or tok in seen:
            continue
        if tok in section_tokens:
            matched.append(tok)
            seen.add(tok)
    return matched


def _current_equivalent_info(section_record: dict, lookup: dict) -> CurrentEquivalentInfo:
    """
    Wraps temporal_filter.get_current_equivalent with direction-aware
    framing:
      - IPC result -> "what's the current (BNS) provision for this?"
      - BNS result -> "what historical (IPC) provision(s) did this
        consolidate or replace?" (informational, per Session 14's
        current-vs-historical framing)
    """
    if section_record["law"] == "IPC":
        result = get_current_equivalent(section_record, lookup)
        if result["status"] == "no_current_equivalent":
            return CurrentEquivalentInfo(
                direction="ipc_to_bns",
                status="no_current_equivalent",
                message=result["message"],
            )
        equivalents = result["equivalents"]
        if len(equivalents) == 1:
            eq = equivalents[0]
            msg = f"Current provision: BNS Section {eq['section']} ({eq['heading']})."
        else:
            names = ", ".join(f"BNS {e['section']}" for e in equivalents)
            msg = f"Current provision(s): {names}."
        return CurrentEquivalentInfo(
            direction="ipc_to_bns", status="mapped", message=msg, sections=equivalents,
        )

    # BNS record: look up which historical IPC section(s) mapped onto it.
    mapped = section_record.get("mapped_sections", [])
    if not mapped:
        return CurrentEquivalentInfo(
            direction="bns_to_ipc",
            status="no_predecessor",
            message="This is a new provision under BNS with no direct IPC predecessor.",
        )
    predecessors = [lookup[("IPC", s)] for s in mapped if ("IPC", s) in lookup]
    if len(predecessors) == 1:
        p = predecessors[0]
        msg = f"Historical provision: IPC Section {p['section']} ({p['heading']})."
    else:
        names = ", ".join(f"IPC {p['section']}" for p in predecessors)
        msg = (f"This provision consolidates {len(predecessors)} former IPC sections: {names}.")
    return CurrentEquivalentInfo(
        direction="bns_to_ipc", status="mapped", message=msg, sections=predecessors,
    )


def explain_result(
    hybrid_result,             # hybrid_ranker.HybridResult
    resolution: LawResolution,
    query: str,
    lookup: dict,               # temporal_filter.build_lookup_index(all_sections)
) -> Explanation:
    record = hybrid_result.section_record
    return Explanation(
        law=record["law"],
        section=record["section"],
        heading=record["heading"],
        full_text=record["text"],
        law_reason=resolution.reason,
        law_confidence=resolution.confidence,
        law_warnings=list(resolution.warnings),
        bm25_score_raw=hybrid_result.bm25_score_raw,
        bm25_score_norm=hybrid_result.bm25_score_norm,
        semantic_score_raw=hybrid_result.semantic_score_raw,
        semantic_score_norm=hybrid_result.semantic_score_norm,
        fused_score=hybrid_result.fused_score,
        matched_keywords=_matched_keywords(query, record),
        current_equivalent=_current_equivalent_info(record, lookup),
    )


def explain_results(
    hybrid_results: list,       # list[hybrid_ranker.HybridResult]
    resolution: LawResolution,
    query: str,
    all_sections: list[dict],
) -> list[Explanation]:
    lookup = build_lookup_index(all_sections)
    return [explain_result(r, resolution, query, lookup) for r in hybrid_results]


def format_explanation(exp: Explanation) -> str:
    """Human-readable rendering matching the Session 10 mock-up format."""
    lines = [
        f"WHY WAS {exp.law} SECTION {exp.section} SELECTED?",
        "",
        f"Law: {exp.law}",
        f"Reason: {exp.law_reason}",
    ]
    if exp.law_warnings:
        lines.append("Warnings:")
        for w in exp.law_warnings:
            lines.append(f"  - {w}")
    lines += [
        "",
        f"BM25 score:      raw={exp.bm25_score_raw:.3f}  normalized={exp.bm25_score_norm:.3f}",
        f"Semantic score:  raw={exp.semantic_score_raw:.3f}  normalized={exp.semantic_score_norm:.3f}",
        f"Combined score:  {exp.fused_score:.3f}",
        "",
        f"Matched concepts: {', '.join(exp.matched_keywords) if exp.matched_keywords else '(none detected)'}",
        "",
        exp.current_equivalent.message,
        "",
        f"Note: {exp.disclaimer}",
    ]
    return "\n".join(lines)
