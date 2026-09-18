"""
Phase 5 — Hybrid Ranking (Score Fusion)
------------------------------------------
Combines Phase 3 (BM25) and Phase 4 (semantic/Qdrant) results into a
single ranked list.

Key design decisions, worth understanding rather than just running:

1. BM25 and semantic scores live on completely different, incomparable
   scales (BM25 is an unbounded term-weighting score, cosine similarity
   is bounded [-1, 1] / [0, 1] after normalize_embeddings). You cannot
   fuse raw scores directly — they must be normalized to a common
   [0, 1] scale first. This module uses min-max normalization over the
   *retrieved candidate set* (i.e. relative to what actually came back
   for this query), which is the standard, defensible choice for
   query-time fusion — you don't have access to global score
   distributions at query time, only what this query retrieved.

2. BM25 and semantic search won't always retrieve the same top-K
   sections. A section that appears in one ranked list but not the
   other is NOT penalized to zero automatically — that would unfairly
   punish sections that one retriever missed but the other found
   strongly. Instead we treat a "missing" score as the minimum observed
   score in that list (i.e. as if it had ranked last, not scored zero
   raw). This is a deliberate, documented choice — flag it in your
   paper's methodology section, since it's a real design decision that
   affects results, not an invisible default.

3. alpha (α) controls the BM25-vs-semantic weighting:
   final = α · BM25_norm + (1-α) · Semantic_norm
   α=1.0 is pure keyword search, α=0.0 is pure semantic search.
   Sweeping α and comparing against gold answers is a Phase 9 job
   (needs the eval set) — this module only provides the mechanism.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class HybridResult:
    section_record: dict
    bm25_score_raw: float
    bm25_score_norm: float
    semantic_score_raw: float
    semantic_score_norm: float
    fused_score: float


def _key(record: dict) -> tuple[str, str]:
    return (record["law"], record["section"])


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        # all identical scores (or a single result) -> treat as fully relevant,
        # avoids a divide-by-zero and avoids arbitrarily zeroing out a lone hit
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fuse_results(
    bm25_results: list,   # list[bm25_index.BM25Result]
    semantic_results: list,  # list[semantic_index.SemanticResult]
    alpha: float = 0.3,
    top_k: int = 10,
) -> list[HybridResult]:
    """
    Normalizes each result list independently (min-max over what was
    retrieved), then fuses by section identity. Sections appearing in
    only one list get that list's minimum normalized score as their
    "missing" value in the other list (see module docstring, point 2).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    bm25_by_key = {_key(r.section_record): r for r in bm25_results}
    sem_by_key = {_key(r.section_record): r for r in semantic_results}

    bm25_raw = [r.score for r in bm25_results]
    sem_raw = [r.score for r in semantic_results]
    bm25_norm_list = _minmax_normalize(bm25_raw)
    sem_norm_list = _minmax_normalize(sem_raw)

    bm25_norm_by_key = {_key(r.section_record): n for r, n in zip(bm25_results, bm25_norm_list)}
    sem_norm_by_key = {_key(r.section_record): n for r, n in zip(semantic_results, sem_norm_list)}

    # "Missing from this list" fallback score = min normalized score seen
    # in that list (or 0.0 if the list was empty entirely).
    bm25_floor = min(bm25_norm_list) if bm25_norm_list else 0.0
    sem_floor = min(sem_norm_list) if sem_norm_list else 0.0

    all_keys = set(bm25_by_key) | set(sem_by_key)

    fused = []
    for key in all_keys:
        bm25_r = bm25_by_key.get(key)
        sem_r = sem_by_key.get(key)

        record = bm25_r.section_record if bm25_r else sem_r.section_record
        bm25_raw_score = bm25_r.score if bm25_r else 0.0
        sem_raw_score = sem_r.score if sem_r else 0.0
        bm25_norm = bm25_norm_by_key.get(key, bm25_floor)
        sem_norm = sem_norm_by_key.get(key, sem_floor)

        fused_score = alpha * bm25_norm + (1 - alpha) * sem_norm

        fused.append(HybridResult(
            section_record=record,
            bm25_score_raw=bm25_raw_score,
            bm25_score_norm=bm25_norm,
            semantic_score_raw=sem_raw_score,
            semantic_score_norm=sem_norm,
            fused_score=fused_score,
        ))

    fused.sort(key=lambda r: (r.fused_score, r.section_record["law"], r.section_record["section"]), reverse=True)
    return fused[:top_k]


def hybrid_search_with_temporal_filter(
    bm25_index,
    semantic_index,
    all_sections: list[dict],
    query: str,
    incident_date,
    alpha: float = 0.3,
    top_k: int = 10,
    retrieval_k: int = 30,
):
    """
    Full retrieval pipeline:

        Phase 2 -> Temporal filtering
        Phase 3 -> BM25
        Phase 4 -> Semantic
        Phase 5 -> Hybrid fusion
        Phase 13 -> Section-aware reranking

    Important:
        Fusion is performed over the wider retrieval_k candidate pool.

        The section-aware reranker then receives that wider pool and
        selects the final top_k results.

    Example:

        retrieval_k = 30
        top_k = 5

        BM25       -> 30
        Semantic   -> 30
        Hybrid     -> up to 60 unique candidates
        Reranker   -> 5 final candidates
    """

    import bm25_index as bm25_mod
    import semantic_index as sem_mod

    from section_reranker import rerank_sections

    # ---------------------------------------------------------
    # Phase 3 — BM25
    # ---------------------------------------------------------

    bm25_results, resolution = (
        bm25_mod.search_with_temporal_filter(
            bm25_index,
            all_sections,
            query,
            incident_date,
            top_k=retrieval_k,
        )
    )

    # ---------------------------------------------------------
    # Phase 4 — Semantic
    # ---------------------------------------------------------

    semantic_results, _ = (
        sem_mod.search_with_temporal_filter(
            semantic_index,
            all_sections,
            query,
            incident_date,
            top_k=retrieval_k,
        )
    )

    # ---------------------------------------------------------
    # Phase 5 — Hybrid fusion
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Use retrieval_k here, NOT top_k.
    #
    # We want the reranker to see the wider candidate pool.
    #

    fused = fuse_results(
        bm25_results,
        semantic_results,
        alpha=alpha,
        top_k=retrieval_k,
    )

    # ---------------------------------------------------------
    # Phase 13 — Section-aware reranking
    # ---------------------------------------------------------

    reranked = rerank_sections(
        query=query,
        hybrid_results=fused,
        top_k=top_k,
    )

    # ---------------------------------------------------------
    # Convert back to HybridResult
    # ---------------------------------------------------------
    #
    # Existing Phase 6/7/10 code expects HybridResult-like
    # objects.
    #
    # Therefore we preserve the original HybridResult objects
    # instead of changing the downstream architecture.
    #

    final_results = [
        item.hybrid_result
        for item in reranked
    ]

    return final_results, resolution


def sweep_alpha(
    bm25_results,
    semantic_results,
    alphas: list[float] = (0.3, 0.5, 0.7),
    top_k: int = 10,
) -> dict[float, list[HybridResult]]:
    """
    Runs fusion at each alpha on the *same* pre-retrieved BM25/semantic
    result lists (avoids re-querying the indices per alpha — cheap and
    correct, since normalization only depends on the retrieved sets).

    NOTE: this only produces ranked lists per alpha. Deciding which
    alpha "wins" requires scoring each against gold answers (Hit@1,
    Hit@3, MRR) — that comparison belongs in Phase 9 once the eval
    set (Phase 8) exists. This function is the mechanism Phase 9 will
    call per question.
    """
    return {
        alpha: fuse_results(bm25_results, semantic_results, alpha=alpha, top_k=top_k)
        for alpha in alphas
    }
