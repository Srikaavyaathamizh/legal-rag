"""
SANDBOX-ONLY test — not part of the shipped project.

This sandbox has no internet access to huggingface.co, so the real
sentence-transformers model can't be downloaded here. This script swaps
in a local TF-IDF vectorizer (scikit-learn, pure math, no download) as a
stand-in embedder, purely to prove the Qdrant wiring — collection
creation, upsert, payload metadata filtering (law + candidate_section_ids),
and the temporal-filter integration — all work correctly end to end.

TF-IDF is NOT true semantic search (no synonym/paraphrase understanding),
so don't judge retrieval *quality* from this test. When you run
semantic_index.py locally with real internet access, SentenceTransformer
will download normally and you'll get genuine semantic matching (e.g.
"careless driving caused death" matching "rash or negligent act" even
though the words differ) — that's the actual point of Phase 4, this test
only confirms the surrounding machinery is correct.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny,
)
from temporal_filter import get_candidate_sections

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
with open(DATA_PATH, encoding="utf-8") as f:
    all_sections = json.load(f)["sections"]

records = [s for s in all_sections if s["status"] != "repealed_in_ipc"]
texts = [f"{r['heading']} {r['text']}" for r in records]

print("Fitting local TF-IDF stand-in (no download needed)...")
vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts).toarray()
tfidf_matrix = sk_normalize(tfidf_matrix)  # so Qdrant cosine distance behaves like real embeddings
VECTOR_SIZE = tfidf_matrix.shape[1]
print(f"Vector size: {VECTOR_SIZE}, corpus size: {len(records)}")

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="legal_sections_test",
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

lookup = {}
points = []
for idx, (record, vector) in enumerate(zip(records, tfidf_matrix)):
    lookup[idx] = record
    points.append(PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={"law": record["law"], "section": record["section"], "status": record["status"]},
    ))
client.upsert(collection_name="legal_sections_test", points=points)
print("Upserted", len(points), "points into in-memory Qdrant collection.\n")


def query_vector_for(text: str):
    v = vectorizer.transform([text]).toarray()[0]
    v = sk_normalize([v])[0]
    return v.tolist()


def search(query, law=None, candidate_section_ids=None, top_k=5):
    must = []
    if law is not None:
        must.append(FieldCondition(key="law", match=MatchValue(value=law)))
    if candidate_section_ids is not None:
        must.append(FieldCondition(key="section", match=MatchAny(any=list(candidate_section_ids))))
    qf = Filter(must=must) if must else None
    response = client.query_points(
        collection_name="legal_sections_test",
        query=query_vector_for(query),
        query_filter=qf,
        limit=top_k,
    )
    return [(lookup[h.id], h.score) for h in response.points]


print("=" * 70)
print("TEST 1 — law='IPC' metadata filter only, no temporal restriction")
for rec, score in search("death negligence rash act", law="IPC"):
    print(f"  {score:.3f}  IPC {rec['section']:>6}  {rec['heading']}")

print("=" * 70)
print("TEST 2 — law='BNS' metadata filter only")
for rec, score in search("death negligence rash act", law="BNS"):
    print(f"  {score:.3f}  BNS {rec['section']:>6}  {rec['heading']}")

print("=" * 70)
print("TEST 3 — Full temporal integration: incident_date=2023-06-01 (pre-cutover)")
candidates, resolution = get_candidate_sections(all_sections, date(2023, 6, 1))
candidate_ids = {s["section"] for s in candidates}
print("Resolved law:", resolution.law, "| candidate pool size:", len(candidate_ids))
for rec, score in search("death negligence rash act", law=resolution.law, candidate_section_ids=candidate_ids):
    print(f"  {score:.3f}  {rec['law']} {rec['section']:>6}  {rec['heading']}")

print("=" * 70)
print("TEST 4 — Full temporal integration: incident_date=2025-01-01 (post-cutover)")
candidates, resolution = get_candidate_sections(all_sections, date(2025, 1, 1))
candidate_ids = {s["section"] for s in candidates}
print("Resolved law:", resolution.law, "| candidate pool size:", len(candidate_ids))
for rec, score in search("death negligence rash act", law=resolution.law, candidate_section_ids=candidate_ids):
    print(f"  {score:.3f}  {rec['law']} {rec['section']:>6}  {rec['heading']}")

print("=" * 70)
print("TEST 5 — Verify metadata filter actually excludes the other law")
for rec, score in search("death negligence rash act", law="IPC"):
    assert rec["law"] == "IPC", "FILTER LEAK: got a non-IPC result under an IPC filter!"
print("PASS — no cross-law leakage in filtered results.")

print("=" * 70)
print("TEST 6 — Verify candidate_section_ids restriction actually restricts")
tiny_candidate_set = {"304A", "302"}
results = search("death negligence", law="IPC", candidate_section_ids=tiny_candidate_set, top_k=10)
returned_sections = {rec["section"] for rec, _ in results}
assert returned_sections.issubset(tiny_candidate_set), f"LEAK: got {returned_sections}"
print("PASS — results restricted to:", returned_sections)
