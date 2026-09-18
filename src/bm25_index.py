"""
Phase 3 — BM25 Keyword Retrieval
---------------------------------
Builds a BM25 index over the unified section corpus, split by law
(IPC / BNS), so that at query time we can search only within the
temporally-filtered candidate set from Phase 2 rather than the
full mixed corpus.

Design choice: rather than one BM25 index over everything and filtering
results after ranking, we index once, then at query time restrict the
*search itself* to the candidate section ids. This matters for BM25
specifically because its IDF statistics are computed over whichever
document set you treat as "the corpus" — searching only within the
temporally-correct law and scoring against that law's own corpus
statistics is more faithful to what BM25 is supposed to measure than
mixing IPC and BNS term statistics together.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

from temporal_filter import get_candidate_sections, LawResolution


TOKEN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> list[str]:
    """Simple, transparent tokenizer: lowercase, alphabetic tokens only.

    Deliberately not stemming/lemmatizing yet — keep Phase 3 simple and
    inspectable; revisit only if evaluation in Phase 9 shows it's needed.
    """
    return TOKEN_RE.findall(text.lower())


@dataclass
class BM25Result:
    section_record: dict
    score: float


class LawScopedBM25:
    """
    Holds two independent BM25 indices (IPC, BNS) built once from the
    full corpus.

    search() takes an optional candidate_ids restriction so Phase 2's
    temporal filtering can narrow what's actually searched.
    """

    def __init__(self, all_sections: list[dict]):
        self.all_sections = all_sections

        self._by_law: dict[str, list[dict]] = {
            "IPC": [],
            "BNS": []
        }

        for s in all_sections:
            # Exclude repealed_in_ipc sections.
            if s["status"] == "repealed_in_ipc":
                continue

            self._by_law[s["law"]].append(s)

        self._bm25: dict[str, BM25Okapi] = {}
        self._tokenized_corpus: dict[str, list[list[str]]] = {}

        for law, records in self._by_law.items():
            corpus_texts = [
                f"{r['heading']} {r['text']}"
                for r in records
            ]

            tokenized = [
                tokenize(t)
                for t in corpus_texts
            ]

            self._tokenized_corpus[law] = tokenized
            self._bm25[law] = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        law: str,
        top_k: int = 10,
        candidate_section_ids: Optional[set[str]] = None,
    ) -> list[BM25Result]:
        """
        Search within a single law's index.

        candidate_section_ids:
            If provided, restrict results to these section identifiers.

        Scores still come from the full-law BM25 model so that IDF
        statistics remain consistent within that law.
        """

        if law not in self._bm25:
            raise ValueError(
                f"No BM25 index for law={law!r}"
            )

        query_tokens = tokenize(query)

        scores = self._bm25[law].get_scores(query_tokens)

        records = self._by_law[law]

        scored = list(zip(records, scores))

        if candidate_section_ids is not None:
            scored = [
                (r, sc)
                for r, sc in scored
                if r["section"] in candidate_section_ids
            ]

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            BM25Result(
                section_record=r,
                score=float(sc)
            )
            for r, sc in scored[:top_k]
        ]


def search_with_temporal_filter(
    index: "LawScopedBM25",
    all_sections: list[dict],
    query: str,
    incident_date,
    top_k: int = 10,
) -> tuple[list[BM25Result], LawResolution]:
    """
    Convenience wrapper tying Phase 2 + Phase 3 together.

    Resolves the applicable law from the incident date, then performs
    BM25 search only within that law's candidate sections.
    """

    candidates, resolution = get_candidate_sections(
        all_sections,
        incident_date
    )

    if resolution.law == "AMBIGUOUS":
        # No date -> can't pick a single law index.
        # Search both laws separately.

        candidate_ids = {
            s["section"]
            for s in candidates
        }

        ipc_results = index.search(
            query,
            "IPC",
            top_k=top_k,
            candidate_section_ids=candidate_ids
        )

        bns_results = index.search(
            query,
            "BNS",
            top_k=top_k,
            candidate_section_ids=candidate_ids
        )

        # BM25 scores from separate corpora are not directly comparable.
        # Therefore interleave results rather than globally sorting
        # their raw scores.

        interleaved = []

        for pair in zip(ipc_results, bns_results):
            interleaved.extend(pair)

        interleaved.extend(
            ipc_results[len(bns_results):]
        )

        interleaved.extend(
            bns_results[len(ipc_results):]
        )

        return interleaved, resolution

    candidate_ids = {
        s["section"]
        for s in candidates
    }

    results = index.search(
        query,
        resolution.law,
        top_k=top_k,
        candidate_section_ids=candidate_ids
    )

    return results, resolution


def load_corpus(
    path: str = "data/processed/unified_sections.json"
) -> list[dict]:
    """
    Load the unified legal corpus.

    UTF-8 is explicitly specified because Windows may otherwise
    default to cp1252 and fail when reading UTF-8 JSON containing
    special Unicode characters.
    """

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data["sections"]