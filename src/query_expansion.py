"""
Query Expansion for Known Legal Abbreviations / Colloquialisms
------------------------------------------------------------------
Small, hand-curated dictionary only -- deliberately NOT an automatically
generated synonym list. A large auto-generated legal-synonym dictionary
risks introducing bias or false-equivalence relationships (see project
notes on avoiding overfitting to a handful of examples). This starts
with a handful of terms known to fail from the UI transcript testing,
and should be grown one verified entry at a time, not bulk-generated.

Usage:
    from query_expansion import expand_query
    expanded = expand_query("POCSO")
    # -> "POCSO child sexual offences protection of children from
    #     sexual offences"

Wire this in at the point where the raw user query is first received,
BEFORE it's passed to BM25/semantic retrieval -- i.e. in pipeline.py or
api/main.py, wherever `query` is first read from the request body.
"""

from __future__ import annotations

# Each entry: short user-typed term -> additional terms appended to the
# query to help retrieval, WITHOUT replacing the original text (so BM25
# exact-match on the original phrase still works too).
#
# Add new entries ONLY after confirming via a real failing query that
# expansion actually helps -- test each addition individually, the same
# discipline used for the alpha-sweep and fusion-strategy experiments.
KNOWN_EXPANSIONS: dict[str, str] = {
    "pocso": "child sexual offences protection of children from sexual offences",
    "drank and drive": "driving under influence of alcohol intoxication",
    "drunk driving": "driving under influence of alcohol intoxication",
    "dui": "driving under influence of alcohol intoxication",
    "cyber harassment": "stalking sexual harassment electronic communication online",
    "cyberbullying": "stalking harassment electronic communication online",
    "child sexual abuse": "sexual offences against children exploitation",
    "chain snatching": "theft snatching robbery",
    "chain chasing": "theft snatching robbery",
    "child marriage": "marrying a child below the age of marriage prohibition of child marriage",
    "acid attack": "voluntarily causing grievous hurt by use of acid",
}


def expand_query(query: str) -> str:
    """
    Case-insensitive substring match against KNOWN_EXPANSIONS. If the
    user's query contains a known term, appends the expansion terms to
    the END of the original query (never replaces it) -- so exact BM25
    term matches on what the user actually typed still work, and the
    appended terms give the semantic/BM25 retrievers additional
    vocabulary to match against.

    Multiple matches can each contribute their expansion (e.g. a query
    containing both "child" and "cyber harassment" gets both).
    """
    query_lower = query.lower()
    additions = []
    for term, expansion in KNOWN_EXPANSIONS.items():
        if term in query_lower:
            additions.append(expansion)

    if not additions:
        return query

    return f"{query} {' '.join(additions)}"
