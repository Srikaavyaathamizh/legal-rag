"""
SANDBOX-ONLY semantic backend — NOT part of the shipped project.

Same rationale as tests/test_semantic_index_sandbox_wiring.py from
Phase 4: this offline sandbox cannot download sentence-transformers
model weights from huggingface.co. This class exposes the exact same
.search(query, law=None, candidate_section_ids=None, top_k=10)
interface as semantic_index.SemanticIndex, so run_ablation.py's system
functions work identically whether given this stub or the real thing —
no code in run_ablation.py needs to change to run for real locally.

TF-IDF is NOT semantic search (no synonym/paraphrase understanding).
Ablation numbers produced with this stub show whether the *retrieval
architecture and metric pipeline* work correctly end-to-end; they are
NOT the numbers that belong in a paper. Re-run run_ablation.py locally
with real_semantic_index() swapped in (see bottom of this file) for
reportable results.
"""

from __future__ import annotations
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class _FakeSemanticResult:
    section_record: dict
    score: float


class TfidfSemanticStub:
    def __init__(self, all_sections: list[dict]):
        self.records = [s for s in all_sections if s["status"] != "repealed_in_ipc"]
        texts = [f"{r['heading']} {r['text']}" for r in self.records]
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        self.matrix = sk_normalize(self.vectorizer.fit_transform(texts).toarray())

    def search(self, query, law=None, candidate_section_ids=None, top_k=10):
        qvec = sk_normalize(self.vectorizer.transform([query]).toarray())
        sims = cosine_similarity(qvec, self.matrix)[0]
        scored = list(zip(self.records, sims))
        if law is not None:
            scored = [(r, s) for r, s in scored if r["law"] == law]
        if candidate_section_ids is not None:
            scored = [(r, s) for r, s in scored if r["section"] in candidate_section_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [_FakeSemanticResult(r, float(s)) for r, s in scored[:top_k]]


def sandbox_semantic_index(all_sections: list[dict]) -> TfidfSemanticStub:
    """Sandbox default — used because huggingface.co is unreachable here."""
    return TfidfSemanticStub(all_sections)


def real_semantic_index(all_sections: list[dict]):
    """
    What you should actually call when running this locally with
    internet access. Requires: pip install sentence-transformers qdrant-client
    """
    from semantic_index import SemanticIndex
    idx = SemanticIndex()
    idx.build(all_sections)
    return idx
