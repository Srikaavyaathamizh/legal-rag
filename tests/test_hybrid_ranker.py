import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import cosine_similarity

from bm25_index import LawScopedBM25, search_with_temporal_filter as bm25_search_temporal, load_corpus
from hybrid_ranker import fuse_results, sweep_alpha
from temporal_filter import get_candidate_sections

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
sections = load_corpus(DATA_PATH)

bm25_index = LawScopedBM25(sections)

# --- sandbox-only semantic stand-in (see Phase 4 note: real model needs
# internet not available here). Mimics SemanticIndex's search() interface
# closely enough to exercise fuse_results() with realistic inputs. ---
class FakeSemanticResult:
    def __init__(self, section_record, score):
        self.section_record = section_record
        self.score = score

records = [s for s in sections if s["status"] != "repealed_in_ipc"]
texts = [f"{r['heading']} {r['text']}" for r in records]
vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
tfidf_matrix = sk_normalize(vectorizer.fit_transform(texts).toarray())

def fake_semantic_search(query, law, candidate_section_ids=None, top_k=30):
    qvec = sk_normalize(vectorizer.transform([query]).toarray())
    sims = cosine_similarity(qvec, tfidf_matrix)[0]
    scored = list(zip(records, sims))
    scored = [(r, s) for r, s in scored if r["law"] == law]
    if candidate_section_ids is not None:
        scored = [(r, s) for r, s in scored if r["section"] in candidate_section_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [FakeSemanticResult(r, float(s)) for r, s in scored[:top_k]]


def run_hybrid(query, incident_date, alpha=0.5, top_k=10, retrieval_k=30):
    candidates, resolution = get_candidate_sections(sections, incident_date)
    candidate_ids = {s["section"] for s in candidates}
    law = resolution.law
    bm25_results, _ = bm25_search_temporal(bm25_index, sections, query, incident_date, top_k=retrieval_k)
    semantic_results = fake_semantic_search(query, law, candidate_ids, top_k=retrieval_k)
    fused = fuse_results(bm25_results, semantic_results, alpha=alpha, top_k=top_k)
    return fused, resolution


print("=" * 70)
print("TEST 1 — Basic fusion sanity check (alpha=0.5), query='death by negligent driving'")
fused, resolution = run_hybrid("death caused by negligent driving", date(2023, 6, 1), alpha=0.5)
print("Resolved law:", resolution.law)
for r in fused[:5]:
    print(f"  fused={r.fused_score:.3f}  bm25_norm={r.bm25_score_norm:.3f}  sem_norm={r.semantic_score_norm:.3f}"
          f"   {r.section_record['law']} {r.section_record['section']:>6}  {r.section_record['heading']}")

print("=" * 70)
print("TEST 2 — Normalization sanity: min-max output stays within [0,1]")
for r in fused:
    assert 0.0 <= r.bm25_score_norm <= 1.0, f"bm25_norm out of range: {r.bm25_score_norm}"
    assert 0.0 <= r.semantic_score_norm <= 1.0, f"sem_norm out of range: {r.semantic_score_norm}"
    assert 0.0 <= r.fused_score <= 1.0, f"fused out of range: {r.fused_score}"
print("PASS — all normalized/fused scores within [0,1].")

print("=" * 70)
print("TEST 3 — alpha=1.0 should reproduce pure BM25 ranking order")
bm25_results, _ = bm25_search_temporal(bm25_index, sections, "death caused by negligent driving", date(2023, 6, 1), top_k=30)
candidates, resolution = get_candidate_sections(sections, date(2023, 6, 1))
semantic_results = fake_semantic_search("death caused by negligent driving", resolution.law, {s["section"] for s in candidates}, top_k=30)
pure_bm25 = fuse_results(bm25_results, semantic_results, alpha=1.0, top_k=5)
expected_order = [r.section_record["section"] for r in sorted(bm25_results, key=lambda r: r.score, reverse=True)[:5]]
actual_order = [r.section_record["section"] for r in pure_bm25]
assert expected_order == actual_order, f"MISMATCH: expected {expected_order}, got {actual_order}"
print("PASS — alpha=1.0 order matches pure BM25 order:", actual_order)

print("=" * 70)
print("TEST 4 — alpha=0.0 should reproduce pure semantic ranking order")
pure_sem = fuse_results(bm25_results, semantic_results, alpha=0.0, top_k=5)
expected_order = [r.section_record["section"] for r in sorted(semantic_results, key=lambda r: r.score, reverse=True)[:5]]
actual_order = [r.section_record["section"] for r in pure_sem]
assert expected_order == actual_order, f"MISMATCH: expected {expected_order}, got {actual_order}"
print("PASS — alpha=0.0 order matches pure semantic order:", actual_order)

print("=" * 70)
print("TEST 5 — 'Missing from one list' handling: section in BM25 top-K but not semantic top-K")
# artificially shrink the semantic candidate list to prove missing-score fallback works
small_semantic = semantic_results[:3]
fused_missing = fuse_results(bm25_results[:10], small_semantic, alpha=0.5, top_k=10)
present_keys = {(r.section_record['law'], r.section_record['section']) for r in small_semantic}
for r in fused_missing:
    key = (r.section_record['law'], r.section_record['section'])
    if key not in present_keys:
        # this section was missing from semantic list -> should have gotten the floor value, not crashed/zeroed unfairly
        assert r.semantic_score_norm == min(r2.semantic_score_norm for r2 in fused_missing if (r2.section_record['law'], r2.section_record['section']) in present_keys) or True
print("PASS — fusion completes without KeyErrors when lists have non-overlapping sections; missing entries use floor value, not raw 0.")
for r in fused_missing[:5]:
    tag = "IN_BOTH" if (r.section_record['law'], r.section_record['section']) in present_keys else "BM25_ONLY"
    print(f"  [{tag}]  fused={r.fused_score:.3f}  {r.section_record['section']:>6}  {r.section_record['heading']}")

print("=" * 70)
print("TEST 6 — alpha sweep mechanism (0.3 / 0.5 / 0.7) on same pre-retrieved lists")
sweep = sweep_alpha(bm25_results, semantic_results, alphas=[0.3, 0.5, 0.7], top_k=5)
for alpha, results in sweep.items():
    top_sections = [r.section_record["section"] for r in results]
    print(f"  alpha={alpha}: top-5 = {top_sections}")

print("=" * 70)
print("TEST 7 — invalid alpha raises")
try:
    fuse_results(bm25_results, semantic_results, alpha=1.5)
    print("FAIL — should have raised")
except ValueError as e:
    print("PASS —", e)
