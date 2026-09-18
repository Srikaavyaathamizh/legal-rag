# Temporal & Explainable Hybrid RAG for Indian Legal Question Answering

A retrieval system for Indian criminal law (IPC / BNS) that resolves **which
version of the law applies** from an incident date *before* retrieving,
rather than searching a single mixed corpus and hoping similarity scores
sort out a legal-version distinction they were never designed to make.

Research question: *does temporal law-version filtering reduce
wrong-Act retrieval errors, compared to retrieval without it?*
See `eval/ablation_summary.csv` for the current answer.

## Project structure

```
data/
  raw/                    Source CSVs (IPC/BNS section text, IPC↔BNS mapping)
  processed/               unified_sections.json — cleaned, merged corpus (Phase 1)
  eval/                    Auto-generated eval candidate pools (Phase 8)

src/
  temporal_filter.py       Resolves applicable law from an incident date (Phase 2)
  bm25_index.py             Keyword retrieval, law-scoped (Phase 3)
  semantic_index.py         Embedding + Qdrant retrieval (Phase 4)
  hybrid_ranker.py           Score fusion: α·BM25 + (1-α)·Semantic (Phase 5)
  explainability.py          Per-result explanation: law reason, scores, current/historical mapping (Phase 6)
  llm_explainer.py           LLM explanation + hallucination guard, pluggable backends (Phase 7)
  pipeline.py                 [not yet built] — chains everything above into one callable

eval/
  build_eval_set.py          Auto-generates eval questions from unified_sections.json (Phase 8)
  verify_eval_set.py          Independent cross-check against the raw mapping CSV
  eval_dataset.json            55 stratified eval records (human `verified` pending)
  run_ablation.py              4-system ablation: Hit@1 / Hit@3 / MRR / Law-Version Error Rate (Phase 9)
  semantic_stub_sandbox.py     TF-IDF stand-in used only where sentence-transformers can't reach huggingface.co
  error_analysis.py            [not yet built]

tests/
  test_*.py                   One test file per src/ module, run independently of each other
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Install only the LLM backend(s) you actually plan to use from `requirements.txt`'s
Phase 7 section — see "LLM backends" below.

## Running tests

Each test file is self-contained and can be run directly:

```bash
cd tests
python test_temporal_filter.py
python test_bm25_index.py
python test_hybrid_ranker.py
python test_explainability.py
python test_semantic_index_sandbox_wiring.py   # Qdrant wiring proof, no internet needed
python test_llm_explainer.py
python test_llm_backends.py
```

## Semantic search: sandbox stub vs. the real thing

`sentence-transformers` needs to download model weights from
`huggingface.co` the first time it runs. In restricted/offline
environments, use `eval/semantic_stub_sandbox.py`'s TF-IDF stand-in
instead — it implements the exact same `.search()` interface as
`SemanticIndex`, so no other code needs to change.

**Numbers produced with the stub are for validating the pipeline only.**
For real, reportable results, swap to the real backend (needs internet,
one line):

```python
# in eval/run_ablation.py's main():
from semantic_stub_sandbox import real_semantic_index   # instead of sandbox_semantic_index
semantic_idx = real_semantic_index(sections)
```

## LLM backends (Phase 7)

`src/llm_explainer.py` supports four interchangeable backends via
`get_llm_backend(name)`:

| Backend | Cost | Setup |
|---|---|---|
| `ollama` (default) | Free, local, no rate limit | Install [Ollama](https://ollama.com), `ollama pull llama3.1` |
| `gemini` | Free tier | `GOOGLE_API_KEY` from [aistudio.google.com](https://aistudio.google.com/apikey) |
| `groq` | Free tier | `GROQ_API_KEY` from [console.groq.com](https://console.groq.com/keys) |
| `anthropic` | Paid | `ANTHROPIC_API_KEY` |

```python
from llm_explainer import generate_explanation, get_llm_backend

result = generate_explanation(query, incident_date, evidence,
                               llm_fn=get_llm_backend("gemini"))
```

Every response is checked post-hoc by `detect_uncited_sections()` — an
independent, deterministic check that flags any section number the
model mentions but was never actually given as evidence, regardless of
whether the system prompt was followed.

## Running the ablation (Phase 9)

```bash
cd eval
python run_ablation.py
```

Produces `ablation_results.csv` (per-record, every system × every
question) and `ablation_summary.csv` (aggregated Hit@1/Hit@3/MRR/
Law-Version Error Rate, overall and per eval category).

**Headline finding so far:** temporal filtering doesn't just improve
ranking — it structurally guarantees a 0% law-version error rate
(systems 3 & 4), versus up to 100% for non-temporal baselines on the
`BNS_STANDARD` and `CONSOLIDATED` eval categories. See
`ablation_summary.csv` for the full breakdown; re-run with the real
semantic backend before citing exact Hit@1/Hit@3/MRR figures.

## Data notes

- 1,019 total sections (562 IPC + 457 BNS) unified into
  `data/processed/unified_sections.json`, each tagged with law, status,
  effective dates, and IPC↔BNS cross-references.
- **BNS cutover date: 2024-07-01** (single hard boundary, not a series of
  per-section amendments — see `temporal_filter.py`'s module docstring).
- 5 IPC sections were already dead *within* IPC before BNS existed
  (`status: "repealed_in_ipc"`) — excluded from retrieval, flagged
  separately since their internal repeal dates aren't in this dataset.
- 29 IPC sections have **no** BNS equivalent (repealed without
  replacement); 40 BNS sections **consolidate** more than one IPC
  predecessor (up to 9-to-1).

## Status

| Phase | Status |
|---|---|
| 1 — Data cleaning & unification | ✅ |
| 2 — Temporal filtering | ✅ |
| 3 — BM25 retrieval | ✅ |
| 4 — Semantic retrieval | ✅ (needs local run w/ real embeddings to confirm) |
| 5 — Hybrid ranking | ✅ |
| 6 — Explainability | ✅ |
| 7 — LLM explanation + hallucination guard | ✅ |
| 8 — Evaluation dataset | ✅ auto-generated + auto-checked; human verification pending |
| 9 — Ablation experiments | ✅ ran on sandbox stub; needs re-run w/ real embeddings |
| 9 — Error analysis | ⏳ not yet built |
| — pipeline.py (end-to-end callable) | ⏳ not yet built |
| — API / frontend | ⏳ future work, not required for the research claim |
