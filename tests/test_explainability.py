import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import cosine_similarity

from bm25_index import LawScopedBM25, search_with_temporal_filter as bm25_search_temporal, load_corpus
from hybrid_ranker import fuse_results
from temporal_filter import get_candidate_sections, build_lookup_index
from explainability import explain_results, format_explanation

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "unified_sections.json")
sections = load_corpus(DATA_PATH)
lookup = build_lookup_index(sections)
bm25_index = LawScopedBM25(sections)

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
    scored = [(r, s) for r, s in zip(records, sims) if r["law"] == law]
    if candidate_section_ids is not None:
        scored = [(r, s) for r, s in scored if r["section"] in candidate_section_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [FakeSemanticResult(r, float(s)) for r, s in scored[:top_k]]

def run_pipeline(query, incident_date, alpha=0.5, top_k=5):
    candidates, resolution = get_candidate_sections(sections, incident_date)
    candidate_ids = {s["section"] for s in candidates}
    bm25_results, _ = bm25_search_temporal(bm25_index, sections, query, incident_date, top_k=30)
    semantic_results = fake_semantic_search(query, resolution.law, candidate_ids, top_k=30)
    fused = fuse_results(bm25_results, semantic_results, alpha=alpha, top_k=top_k)
    explanations = explain_results(fused, resolution, query, sections)
    return explanations, resolution


print("=" * 70)
print("TEST 1 — IPC 304A case: full explanation, expect clean current-equivalent (BNS 106)")
explanations, resolution = run_pipeline("death caused by negligent driving", date(2023, 6, 1))
top = explanations[0]
print(format_explanation(top))
assert top.section == "304A", f"expected 304A on top, got {top.section}"
assert top.current_equivalent.status == "mapped"
assert top.current_equivalent.sections[0]["section"] == "106"
assert "death" in top.matched_keywords and "negligent" in " ".join(top.matched_keywords) or "negligent" in top.matched_keywords
print("PASS")

print("=" * 70)
print("TEST 2 — BNS-side query (post-cutover date): expect historical IPC provision surfaced")
explanations, resolution = run_pipeline("death caused by negligent driving", date(2025, 1, 1))
top = explanations[0]
print(format_explanation(top))
assert top.law == "BNS"
assert top.current_equivalent.direction == "bns_to_ipc"
assert top.current_equivalent.status == "mapped"
print("PASS")

print("=" * 70)
print("TEST 3 — Repealed-without-replacement section: explicit query for IPC Section 15 by heading text")
# Section 15/16 have empty text (repealed_in_ipc, excluded from indices) so we
# instead force-explain a genuinely repealed-WITH-mapping=REPEALED section that
# still has real text: search for one from the ['REPEALED'] set found earlier.
repealed_example = next(s for s in sections if s["law"] == "IPC" and s.get("mapped_sections") == ["REPEALED"] and s["text"])
print("Using:", repealed_example["section"], repealed_example["heading"])
from hybrid_ranker import HybridResult
fake_hybrid = HybridResult(
    section_record=repealed_example,
    bm25_score_raw=10.0, bm25_score_norm=1.0,
    semantic_score_raw=0.5, semantic_score_norm=1.0,
    fused_score=1.0,
)
resolution_ipc = get_candidate_sections(sections, date(2023,1,1))[1]
exp = explain_results([fake_hybrid], resolution_ipc, repealed_example["heading"], sections)[0]
print(format_explanation(exp))
assert exp.current_equivalent.status == "no_current_equivalent"
print("PASS — correctly reports no current BNS equivalent")

print("=" * 70)
print("TEST 4 — Many-to-one consolidation: BNS 127 should list all 9 historical IPC sections")
bns_127 = next(s for s in sections if s["law"] == "BNS" and s["section"] == "127")
fake_hybrid_127 = HybridResult(
    section_record=bns_127,
    bm25_score_raw=5.0, bm25_score_norm=0.8,
    semantic_score_raw=0.4, semantic_score_norm=0.8,
    fused_score=0.8,
)
resolution_bns = get_candidate_sections(sections, date(2025,1,1))[1]
exp = explain_results([fake_hybrid_127], resolution_bns, "wrongful confinement", sections)[0]
print(format_explanation(exp))
assert exp.current_equivalent.status == "mapped"
assert len(exp.current_equivalent.sections) == 9, f"expected 9 predecessors, got {len(exp.current_equivalent.sections)}"
print("PASS — correctly lists all 9 consolidated IPC predecessors")

print("=" * 70)
print("TEST 5 — Matched keywords excludes stopwords and stays in query order, deduplicated")
explanations, resolution = run_pipeline("the death of the person by the negligent act", date(2023, 6, 1))
top_kw = explanations[0].matched_keywords
print("matched_keywords:", top_kw)
assert "the" not in top_kw and "of" not in top_kw and "by" not in top_kw
assert len(top_kw) == len(set(top_kw)), "duplicates found"
print("PASS")
