import sys
import os
from datetime import date

# Add src folder to Python path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from bm25_index import (
    LawScopedBM25,
    search_with_temporal_filter,
    load_corpus,
)


# Path to unified sections JSON
data_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "unified_sections.json"
)


# Load corpus
sections = load_corpus(data_file)

# Build BM25 index
index = LawScopedBM25(sections)


# ============================================================
# TEST 1
# ============================================================

print("=" * 70)
print("TEST 1 — Direct BM25 search, IPC index, no temporal restriction")
print("Query: 'death negligence'")

for r in index.search(
    "death negligence",
    law="IPC",
    top_k=5
):
    print(
        f"  {r.score:.3f}  "
        f"IPC {r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 2
# ============================================================

print("=" * 70)
print("TEST 2 — Same query, BNS index")

for r in index.search(
    "death negligence",
    law="BNS",
    top_k=5
):
    print(
        f"  {r.score:.3f}  "
        f"BNS {r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 3
# ============================================================

print("=" * 70)
print(
    "TEST 3 — Full pipeline: "
    "query + incident_date=2023-12-10 "
    "(pre-cutover) -> should search IPC only"
)

results, resolution = search_with_temporal_filter(
    index,
    sections,
    "causing death by negligent driving",
    date(2023, 12, 10),
    top_k=5
)

print("Resolved law:", resolution.law)

for r in results:
    print(
        f"  {r.score:.3f}  "
        f"{r.section_record['law']} "
        f"{r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 4
# ============================================================

print("=" * 70)
print(
    "TEST 4 — Same query, "
    "incident_date=2025-01-01 "
    "(post-cutover) -> should search BNS only"
)

results, resolution = search_with_temporal_filter(
    index,
    sections,
    "causing death by negligent driving",
    date(2025, 1, 1),
    top_k=5
)

print("Resolved law:", resolution.law)

for r in results:
    print(
        f"  {r.score:.3f}  "
        f"{r.section_record['law']} "
        f"{r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 5
# ============================================================

print("=" * 70)
print(
    "TEST 5 — No date given -> AMBIGUOUS, "
    "should return results from BOTH laws"
)

results, resolution = search_with_temporal_filter(
    index,
    sections,
    "causing death by negligent driving",
    None,
    top_k=3
)

print(
    "Resolved law:",
    resolution.law,
    "| total results:",
    len(results)
)

for r in results:
    print(
        f"  {r.score:.3f}  "
        f"{r.section_record['law']} "
        f"{r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 6
# ============================================================

print("=" * 70)
print(
    "TEST 6 — Sanity check: "
    "cheating query should surface IPC 420 / "
    "its BNS equivalent"
)

results, resolution = search_with_temporal_filter(
    index,
    sections,
    "cheating and dishonestly inducing delivery of property",
    date(2023, 1, 1),
    top_k=5
)

print("Resolved law:", resolution.law)

for r in results:
    print(
        f"  {r.score:.3f}  "
        f"{r.section_record['law']} "
        f"{r.section_record['section']:>6}  "
        f"{r.section_record['heading']}"
    )


print("=" * 70)
print("ALL BM25 TESTS COMPLETED")
print("=" * 70)