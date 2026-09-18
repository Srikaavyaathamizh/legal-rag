import sys
import os
import json
from datetime import date

# ============================================================
# Add src/ to Python import path
# ============================================================

SRC_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src"
)

sys.path.insert(0, SRC_PATH)


from semantic_index import (
    SemanticIndex,
    search_with_temporal_filter,
)


# ============================================================
# Load corpus
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "unified_sections.json"
)

with open(DATA_PATH, encoding="utf-8") as f:
    all_sections = json.load(f)["sections"]


print("=" * 70)
print("REAL SEMANTIC INDEX TEST")
print("=" * 70)

print("Corpus sections:", len(all_sections))


# ============================================================
# Create Semantic Index
# ============================================================

print("\nLoading SentenceTransformer model...")

index = SemanticIndex(
    model_name="all-MiniLM-L6-v2",
    qdrant_location=":memory:",
    collection_name="legal_sections_real_test",
)

print("SentenceTransformer loaded successfully.")


# ============================================================
# Build index
# ============================================================

print("\nBuilding Qdrant semantic index...")
print("This may take some time on the first run.")

index.build(
    all_sections,
    batch_size=64
)

print("Semantic index built successfully.")


# ============================================================
# TEST 1 — IPC semantic search
# ============================================================

print("\n" + "=" * 70)
print("TEST 1 — Semantic search: IPC")
print("Query: careless driving caused death")
print("=" * 70)

results = index.search(
    "careless driving caused death",
    law="IPC",
    top_k=5
)

for r in results:
    print(
        f"{r.score:.4f} | "
        f"{r.section_record['law']} "
        f"{r.section_record['section']} | "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 2 — BNS semantic search
# ============================================================

print("\n" + "=" * 70)
print("TEST 2 — Semantic search: BNS")
print("Query: careless driving caused death")
print("=" * 70)

results = index.search(
    "careless driving caused death",
    law="BNS",
    top_k=5
)

for r in results:
    print(
        f"{r.score:.4f} | "
        f"{r.section_record['law']} "
        f"{r.section_record['section']} | "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 3 — Temporal filtering, IPC
# ============================================================

print("\n" + "=" * 70)
print("TEST 3 — Temporal filtering")
print("Incident date: 2023-12-10")
print("Expected law: IPC")
print("=" * 70)

results, resolution = search_with_temporal_filter(
    index,
    all_sections,
    "careless driving caused death",
    date(2023, 12, 10),
    top_k=5
)

print("Resolved law:", resolution.law)

for r in results:
    print(
        f"{r.score:.4f} | "
        f"{r.section_record['law']} "
        f"{r.section_record['section']} | "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 4 — Temporal filtering, BNS
# ============================================================

print("\n" + "=" * 70)
print("TEST 4 — Temporal filtering")
print("Incident date: 2025-01-01")
print("Expected law: BNS")
print("=" * 70)

results, resolution = search_with_temporal_filter(
    index,
    all_sections,
    "careless driving caused death",
    date(2025, 1, 1),
    top_k=5
)

print("Resolved law:", resolution.law)

for r in results:
    print(
        f"{r.score:.4f} | "
        f"{r.section_record['law']} "
        f"{r.section_record['section']} | "
        f"{r.section_record['heading']}"
    )


# ============================================================
# TEST 5 — Candidate section restriction
# ============================================================

print("\n" + "=" * 70)
print("TEST 5 — Candidate section restriction")
print("=" * 70)

candidate_ids = {"304A", "302"}

results = index.search(
    "death caused by negligence",
    law="IPC",
    candidate_section_ids=candidate_ids,
    top_k=10
)

returned_sections = {
    r.section_record["section"]
    for r in results
}

print("Allowed sections:", candidate_ids)
print("Returned sections:", returned_sections)

assert returned_sections.issubset(candidate_ids)

print("PASS — candidate filtering works.")


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("REAL SEMANTIC INDEX TEST COMPLETED")
print("=" * 70)