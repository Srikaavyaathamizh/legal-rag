"""
Phase 13B — Section-Aware Legal Reranker
-----------------------------------------

Transparent reranking layer after hybrid BM25 + semantic retrieval.

Design goals:
    1. Preserve the original hybrid score.
    2. Add heading and lexical signals.
    3. Add a generic legal-relation signal.
    4. Never hard-code evaluation answers or section IDs.
    5. Preserve temporal-law filtering.
    6. Provide diagnostics for API/evaluation analysis.

This module does NOT perform:
    - temporal filtering
    - BM25 retrieval
    - semantic retrieval
    - LLM generation

It only reranks candidates already retrieved by the hybrid system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ============================================================
# TOKENIZATION
# ============================================================

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "under",
    "regarding",
    "provision",
    "legal",
    "section",
    "covers",
    "cover",
}


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class RerankedResult:
    """
    Wrapper around a HybridResult.

    The original HybridResult is preserved.

    Additional fields explain why the reranker changed ranking.
    """

    hybrid_result: object

    lexical_score: float
    heading_score: float
    exact_phrase_score: float
    section_signal: float
    legal_relation_score: float

    rerank_score: float
    final_score: float

    original_rank: int


# ============================================================
# TOKEN HELPERS
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase alphanumeric tokens.
    """

    if not text:
        return []

    return TOKEN_RE.findall(text.lower())


def meaningful_tokens(text: str) -> list[str]:
    """
    Remove stopwords while preserving order.

    Duplicate tokens are removed.
    """

    tokens = tokenize(text)

    result = []
    seen = set()

    for token in tokens:
        if token in STOPWORDS:
            continue

        if token not in seen:
            result.append(token)
            seen.add(token)

    return result


# ============================================================
# BASIC SIMILARITY
# ============================================================

def overlap_score(
    query_tokens: Iterable[str],
    document_tokens: Iterable[str],
) -> float:
    """
    Query-token coverage.

    Returns:
        value in [0, 1]
    """

    query_set = set(query_tokens)
    document_set = set(document_tokens)

    if not query_set:
        return 0.0

    return len(query_set & document_set) / len(query_set)


def ordered_overlap_score(
    query_tokens: list[str],
    document_tokens: list[str],
) -> float:
    """
    Measures how much of the query appears in the same order
    in the document.
    """

    if not query_tokens or not document_tokens:
        return 0.0

    q_index = 0
    matched = 0

    for token in document_tokens:

        if q_index >= len(query_tokens):
            break

        if token == query_tokens[q_index]:
            matched += 1
            q_index += 1

    return matched / len(query_tokens)


# ============================================================
# HEADING SIMILARITY
# ============================================================

def heading_similarity(
    query: str,
    heading: str,
) -> float:
    """
    Compare query concepts with statutory heading.

    Heading receives special importance because legal questions
    frequently paraphrase statutory headings.
    """

    query_tokens = meaningful_tokens(query)
    heading_tokens = meaningful_tokens(heading)

    if not query_tokens or not heading_tokens:
        return 0.0

    coverage = overlap_score(
        query_tokens,
        heading_tokens,
    )

    ordered = ordered_overlap_score(
        query_tokens,
        heading_tokens,
    )

    score = (
        0.75 * coverage
        + 0.25 * ordered
    )

    return min(1.0, max(0.0, score))


# ============================================================
# EXACT PHRASE SIGNAL
# ============================================================

def exact_phrase_similarity(
    query: str,
    heading: str,
) -> float:
    """
    Detect meaningful consecutive phrases.

    Returns a conservative score in [0, 1].
    """

    query_tokens = meaningful_tokens(query)
    heading_tokens = meaningful_tokens(heading)

    if not query_tokens or not heading_tokens:
        return 0.0

    if len(query_tokens) < 2:
        return 0.0

    query_text = " ".join(query_tokens)
    heading_text = " ".join(heading_tokens)

    # Full normalized query phrase occurs in heading.
    if query_text in heading_text:
        return 1.0

    best = 0

    heading_phrases = set(
        tuple(heading_tokens[i:j])
        for i in range(len(heading_tokens))
        for j in range(i + 2, len(heading_tokens) + 1)
    )

    for start in range(len(query_tokens)):

        for end in range(
            start + 2,
            len(query_tokens) + 1,
        ):

            phrase = tuple(
                query_tokens[start:end]
            )

            if phrase in heading_phrases:
                best = max(
                    best,
                    end - start,
                )

    if best == 0:
        return 0.0

    return min(
        1.0,
        best / len(query_tokens),
    )


# ============================================================
# SECTION NUMBER SIGNAL
# ============================================================

SECTION_PATTERN = re.compile(
    r"\b(?:ipc|bns)?\s*"
    r"(\d+[A-Za-z]?"
    r"(?:\(\d+\))?"
    r"(?:\([a-z]\))?"
    r"(?:\s+Explanation)?)\b",
    re.IGNORECASE,
)


def normalize_section_number(section: str) -> str:
    """
    Normalize section identifiers.
    """

    return re.sub(
        r"\s+",
        " ",
        section.strip().lower(),
    )


def extract_section_references(
    text: str,
) -> list[str]:
    """
    Extract explicit section references from the USER QUERY.

    Example:
        "BNS 318(4)" -> ["318(4)"]

    We do not infer section numbers from keywords.
    """

    if not text:
        return []

    matches = SECTION_PATTERN.findall(text)

    return [
        normalize_section_number(match)
        for match in matches
    ]


def section_number_signal(
    query: str,
    section: str,
) -> float:
    """
    Reward only an explicit section reference.

    1.0 = exact explicit section match
    0.0 = no explicit match
    """

    references = extract_section_references(query)

    if not references:
        return 0.0

    candidate = normalize_section_number(section)

    for reference in references:

        if candidate == reference:
            return 1.0

    return 0.0


# ============================================================
# FULL TEXT SIGNAL
# ============================================================

def text_similarity(
    query: str,
    text: str,
) -> float:
    """
    Lightweight concept overlap against statutory text.
    """

    query_tokens = meaningful_tokens(query)
    text_tokens = meaningful_tokens(text)

    return overlap_score(
        query_tokens,
        text_tokens,
    )


# ============================================================
# LEGAL RELATION SIGNAL
# ============================================================

def legal_relation_score(
    query: str,
    heading: str,
    text: str,
) -> float:
    """
    Generic signal for direct legal-concept support.

    This does NOT contain:
        - evaluation IDs
        - expected sections
        - hard-coded answers

    It compares the query against both the heading and
    statutory text.
    """

    query_tokens = meaningful_tokens(query)

    if not query_tokens:
        return 0.0

    heading_tokens = meaningful_tokens(heading)
    text_tokens = meaningful_tokens(text)

    heading_cov = overlap_score(
        query_tokens,
        heading_tokens,
    )

    text_cov = overlap_score(
        query_tokens,
        text_tokens,
    )

    heading_order = ordered_overlap_score(
        query_tokens,
        heading_tokens,
    )

    text_order = ordered_overlap_score(
        query_tokens,
        text_tokens,
    )

    query_phrase = " ".join(query_tokens)
    text_phrase = " ".join(text_tokens)

    full_phrase = (
        1.0
        if (
            len(query_tokens) >= 2
            and query_phrase in text_phrase
        )
        else 0.0
    )

    score = (
        0.15 * heading_cov
        + 0.10 * heading_order
        + 0.35 * text_cov
        + 0.20 * text_order
        + 0.20 * full_phrase
    )

    return min(
        1.0,
        max(0.0, score),
    )


# ============================================================
# MAIN RERANK SCORE
# ============================================================

def calculate_rerank_score(
    query: str,
    hybrid_result,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Calculate transparent reranking signals.

    The reranker combines:

        1. Original hybrid score
        2. Heading similarity
        3. Exact phrase similarity
        4. Full statutory-text similarity
        5. Explicit section-number match
        6. Generic legal-relation similarity

    Design:
        - Hybrid retrieval remains the strongest signal.
        - Full statutory text receives more weight than heading alone.
        - No evaluation IDs or expected answers are hard-coded.
    """

    record = hybrid_result.section_record

    heading = record.get("heading", "")
    text = record.get("text", "")

    # --------------------------------------------------------
    # Signal 1: heading similarity
    # --------------------------------------------------------

    heading_score = heading_similarity(
        query,
        heading,
    )

    # --------------------------------------------------------
    # Signal 2: exact phrase in heading
    # --------------------------------------------------------

    exact_phrase_score = exact_phrase_similarity(
        query,
        heading,
    )

    # --------------------------------------------------------
    # Signal 3: full statutory text
    # --------------------------------------------------------

    text_score = text_similarity(
        query,
        text,
    )

    # --------------------------------------------------------
    # Signal 4: explicit section reference
    # --------------------------------------------------------

    section_signal = section_number_signal(
        query,
        record.get("section", ""),
    )

    # --------------------------------------------------------
    # Signal 5: generic legal relation
    # --------------------------------------------------------

    legal_relation = legal_relation_score(
        query,
        heading,
        text,
    )

    # --------------------------------------------------------
    # Lexical score
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Give statutory body text more importance than heading.
    #
    # This helps questions where the legal concept is inside
    # a subsection/body even when the section heading differs.
    #

    lexical_score = (
        0.30 * heading_score
        + 0.10 * exact_phrase_score
        + 0.60 * text_score
    )

    # --------------------------------------------------------
    # Original hybrid score
    # --------------------------------------------------------

    original_score = float(
        getattr(
            hybrid_result,
            "fused_score",
            0.0,
        )
    )

    # --------------------------------------------------------
    # FINAL RERANK SCORE
    # --------------------------------------------------------
    #
    # Hybrid retrieval remains dominant.
    #
    # 55% original hybrid
    # 30% lexical/body evidence
    #  5% explicit section reference
    # 10% legal relation
    #

    rerank_score = (
        0.55 * original_score
        + 0.30 * lexical_score
        + 0.05 * section_signal
        + 0.10 * legal_relation
    )

    return (
        lexical_score,
        heading_score,
        exact_phrase_score,
        section_signal,
        legal_relation,
        rerank_score,
    )


# ============================================================
# RERANK FUNCTION
# ============================================================

def rerank_sections(
    query: str,
    hybrid_results: list,
    top_k: int = 5,
) -> list[RerankedResult]:
    """
    Rerank hybrid candidates.

    IMPORTANT:
        Pass the wider retrieval pool here, such as 30 candidates,
        rather than only the final top 5.
    """

    if top_k <= 0:
        return []

    if not hybrid_results:
        return []

    reranked = []

    for original_rank, result in enumerate(
        hybrid_results,
        start=1,
    ):

        (
            lexical_score,
            heading_score,
            exact_phrase_score,
            section_signal,
            legal_relation,
            rerank_score,
        ) = calculate_rerank_score(
            query,
            result,
        )

        reranked.append(
            RerankedResult(
                hybrid_result=result,

                lexical_score=lexical_score,
                heading_score=heading_score,
                exact_phrase_score=exact_phrase_score,
                section_signal=section_signal,
                legal_relation_score=legal_relation,

                rerank_score=rerank_score,
                final_score=rerank_score,

                original_rank=original_rank,
            )
        )

    # --------------------------------------------------------
    # Stable deterministic sorting
    # --------------------------------------------------------

    reranked.sort(
        key=lambda item: (
            item.final_score,
            item.hybrid_result.section_record.get(
                "law",
                "",
            ),
            item.hybrid_result.section_record.get(
                "section",
                "",
            ),
        ),
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# CONVERSION HELPERS
# ============================================================

def reranked_to_hybrid_result(
    result: RerankedResult,
):
    """
    Return the original HybridResult.

    Existing downstream code can therefore continue to work.
    """

    return result.hybrid_result


def reranked_to_dict(
    result: RerankedResult,
) -> dict:
    """
    Convert reranked result into a JSON-friendly dictionary.
    """

    hybrid = result.hybrid_result
    record = hybrid.section_record

    return {
        "law": record.get("law"),
        "section": record.get("section"),
        "heading": record.get("heading"),

        "bm25_score_raw": hybrid.bm25_score_raw,
        "bm25_score_norm": hybrid.bm25_score_norm,

        "semantic_score_raw": hybrid.semantic_score_raw,
        "semantic_score_norm": hybrid.semantic_score_norm,

        "original_fused_score": hybrid.fused_score,

        "lexical_score": result.lexical_score,
        "heading_score": result.heading_score,
        "exact_phrase_score": result.exact_phrase_score,
        "section_signal": result.section_signal,
        "legal_relation_score": result.legal_relation_score,

        "rerank_score": result.rerank_score,
        "final_score": result.final_score,

        "original_rank": result.original_rank,
    }


# ============================================================
# SIMPLE UNIT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("PHASE 13B — SECTION-AWARE LEGAL RERANKER")
    print("=" * 75)

    test_cases = [
        (
            "punishment for using a false property mark",
            "Punishment for using a false property mark",
        ),
        (
            "short title commencement and application",
            "Short title, commencement and application",
        ),
        (
            "when right of private defence of body extends to causing death",
            "When such right extends to causing death",
        ),
        (
            "cheating",
            "Cheating and dishonestly inducing delivery of property",
        ),
    ]

    for query, heading in test_cases:

        heading_score = heading_similarity(
            query,
            heading,
        )

        phrase_score = exact_phrase_similarity(
            query,
            heading,
        )

        relation_score = legal_relation_score(
            query,
            heading,
            heading,
        )

        print()
        print("Query:           ", query)
        print("Heading:         ", heading)
        print(
            f"Heading score:   {heading_score:.3f}"
        )
        print(
            f"Phrase score:    {phrase_score:.3f}"
        )
        print(
            f"Relation score:  {relation_score:.3f}"
        )

    print()
    print("=" * 75)
    print("PHASE 13B TEST COMPLETED")
    print("=" * 75)