"""
Phase 10 — End-to-End Legal RAG Pipeline
-----------------------------------------

Connects:

Phase 2  -> Temporal law resolution
Phase 3  -> BM25 retrieval
Phase 4  -> Semantic retrieval + Qdrant
Phase 5  -> Hybrid ranking
Phase 6  -> Explainability
Phase 7  -> Grounded LLM explanation
Phase 10 -> Complete application pipeline

Usage
-----

    pipeline = LegalRAGPipeline()

    result = pipeline.query(
        "Someone died because of negligent driving",
        "2023-12-10"
    )

    print(result)

Or:

    answer = pipeline.answer(
        "Someone died because of negligent driving",
        "2023-12-10"
    )

"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional


from temporal_filter import build_lookup_index

from bm25_index import LawScopedBM25

from semantic_index import SemanticIndex

from hybrid_ranker import (
    hybrid_search_with_temporal_filter,
)

from explainability import explain_results

from llm_explainer import generate_explanation
from query_expansion import expand_query

# ====================================================================
# PATHS
# ====================================================================

SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SRC_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_sections.json"
)

DEFAULT_QDRANT_PATH = (
    PROJECT_ROOT
    / "qdrant_data"
)


# ====================================================================
# LEGAL RAG PIPELINE
# ====================================================================

class LegalRAGPipeline:

    def __init__(
        self,
        data_path: str | Path = DATA_PATH,
        semantic_model: str = "all-MiniLM-L6-v2",
        qdrant_location: str | Path = DEFAULT_QDRANT_PATH,
        alpha: float = 0.3,
        retrieval_k: int = 30,
        top_k: int = 3,
    ):
        """
        Initialize the complete Legal RAG pipeline.

        Parameters
        ----------
        data_path:
            Location of unified_sections.json.

        semantic_model:
            SentenceTransformer embedding model.

        qdrant_location:
            Qdrant storage.

            ":memory:"
                Temporary Qdrant.

            "./qdrant_data"
                Persistent Qdrant.

            Windows paths are handled safely by semantic_index.py.

        alpha:
            BM25 weight in hybrid ranking.

            alpha = 1.0
                Pure BM25.

            alpha = 0.0
                Pure semantic.

            alpha = 0.3
                Equal weighting.

        retrieval_k:
            Number of candidates retrieved from each retriever
            before hybrid fusion.

        top_k:
            Number of final results returned.
        """

        # ------------------------------------------------------------
        # Validate configuration
        # ------------------------------------------------------------

        if not 0.0 <= alpha <= 1.0:

            raise ValueError(
                f"alpha must be between 0 and 1, got {alpha}"
            )

        if retrieval_k <= 0:

            raise ValueError(
                "retrieval_k must be greater than 0."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        # ------------------------------------------------------------
        # Store configuration
        # ------------------------------------------------------------

        self.data_path = Path(
            data_path
        ).resolve()

        self.qdrant_location = str(
            qdrant_location
        )

        self.alpha = alpha

        self.retrieval_k = retrieval_k

        self.top_k = top_k

        # ------------------------------------------------------------
        # Initialization banner
        # ------------------------------------------------------------

        print("=" * 70)

        print(
            "INITIALIZING LEGAL RAG PIPELINE"
        )

        print("=" * 70)

        # ------------------------------------------------------------
        # Validate corpus
        # ------------------------------------------------------------

        if not self.data_path.exists():

            raise FileNotFoundError(
                "Legal corpus not found:\n"
                f"{self.data_path}"
            )

        # ------------------------------------------------------------
        # Load corpus
        # ------------------------------------------------------------

        print(
            "\n[1/4] Loading legal corpus..."
        )

        with open(
            self.data_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if "sections" not in data:

            raise ValueError(
                "Invalid unified_sections.json format. "
                "Expected a top-level 'sections' key."
            )

        self.sections = data["sections"]

        print(
            f"Loaded {len(self.sections)} legal sections."
        )

        # ------------------------------------------------------------
        # Build lookup index
        # ------------------------------------------------------------

        print(
            "Building section lookup index..."
        )

        self.lookup = build_lookup_index(
            self.sections
        )

        print(
            "Section lookup index ready."
        )

        # ------------------------------------------------------------
        # BM25
        # ------------------------------------------------------------

        print(
            "\n[2/4] Building BM25 index..."
        )

        self.bm25_index = LawScopedBM25(
            self.sections
        )

        print(
            "BM25 index ready."
        )

        # ------------------------------------------------------------
        # Semantic index
        # ------------------------------------------------------------

        print(
            "\n[3/4] Loading semantic model..."
        )

        self.semantic_index = SemanticIndex(
            model_name=semantic_model,

            qdrant_location=self.qdrant_location,
        )

        print(
            "Semantic model ready."
        )

        # ------------------------------------------------------------
        # Build / reuse Qdrant index
        # ------------------------------------------------------------

        print(
            "\n[4/4] Preparing semantic Qdrant index..."
        )

        self.semantic_index.build(
            self.sections
        )

        print(
            "Semantic index ready."
        )

        # ------------------------------------------------------------
        # Ready
        # ------------------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "LEGAL RAG PIPELINE READY"
        )

        print(
            "=" * 70
        )

    # =================================================================
    # DATE PARSER
    # =================================================================

    @staticmethod
    def parse_date(
        value,
    ) -> Optional[date]:
        """
        Convert supported date formats into datetime.date.

        Supported:

            None

            date(2023, 12, 10)

            datetime(...)

            "2023-12-10"

        Returns
        -------
        date | None
        """

        if value is None:

            return None

        # datetime must be checked before date because
        # datetime is a subclass of date.
        if isinstance(value, datetime):

            return value.date()

        if isinstance(value, date):

            return value

        if isinstance(value, str):

            value = value.strip()

            if not value:

                return None

            try:

                return datetime.strptime(
                    value,
                    "%Y-%m-%d",
                ).date()

            except ValueError as exc:

                raise ValueError(
                    "Invalid incident_date. "
                    "Expected YYYY-MM-DD."
                ) from exc

        raise TypeError(
            "incident_date must be a date, datetime, "
            "YYYY-MM-DD string, or None."
        )

    # =================================================================
    # QUERY
    # =================================================================

    def query(
        self,
        question: str,
        incident_date=None,
        generate_llm: bool = True,
    ) -> dict:
        """
        Execute the complete Legal RAG pipeline.

        Pipeline:

            Question
                ↓
            Date parsing
                ↓
            Temporal law resolution
                ↓
            BM25 retrieval
                ↓
            Semantic retrieval
                ↓
            Hybrid ranking
                ↓
            Explainability
                ↓
            Grounded LLM
                ↓
            Hallucination guard
                ↓
            Final result
        """

        # ------------------------------------------------------------
        # Validate question
        # ------------------------------------------------------------

        if not isinstance(
            question,
            str,
        ):

            raise TypeError(
                "question must be a string."
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        # ------------------------------------------------------------
        # Parse date
        # ------------------------------------------------------------

        parsed_date = self.parse_date(
            incident_date
        )
        
        
        # ------------------------------------------------------------
        # Query expansion
        # ------------------------------------------------------------

        search_question = expand_query(question)

        # ------------------------------------------------------------
        # Query information
        # ------------------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "LEGAL RAG QUERY"
        )

        print(
            "=" * 70
        )

        print(
            f"Question: {question}"
        )

        print(
            "Incident date: "
            f"{parsed_date.isoformat() if parsed_date else 'not provided'}"
        )

        # ============================================================
        # PHASE 2 + 3 + 4 + 5
        # ============================================================

        print(
            "\n[1/3] Running hybrid retrieval..."
        )

        hybrid_results, resolution = (
            hybrid_search_with_temporal_filter(
                self.bm25_index,

                self.semantic_index,

                self.sections,

                #question,
                
                search_question,

                parsed_date,

                alpha=self.alpha,

                top_k=self.top_k,

                retrieval_k=self.retrieval_k,
            )
        )

        # ------------------------------------------------------------
        # Resolution
        # ------------------------------------------------------------

        print(
            f"Applicable law: {resolution.law}"
        )

        print(
            f"Confidence: {resolution.confidence}"
        )

        if resolution.warnings:

            print(
                "Warnings:"
            )

            for warning in resolution.warnings:

                print(
                    f"  - {warning}"
                )

        # ------------------------------------------------------------
        # Retrieval results
        # ------------------------------------------------------------

        print(
            f"Retrieved results: "
            f"{len(hybrid_results)}"
        )

        for result in hybrid_results:

            record = result.section_record

            print(
                f"  {record['law']} "
                f"Section {record['section']} "
                f"| fused={result.fused_score:.3f}"
            )

        # ============================================================
        # PHASE 6
        # ============================================================

        print(
            "\n[2/3] Building explanations..."
        )

        explanations = explain_results(
            hybrid_results,

            resolution,

            question,

            self.sections,
        )

        for explanation in explanations:

            print(
                f"  {explanation.law} "
                f"Section {explanation.section} "
                f"| fused={explanation.fused_score:.3f}"
            )

        # ============================================================
        # PHASE 7
        # ============================================================

        llm_result = None

        if generate_llm:

            print(
                "\n[3/3] Generating grounded "
                "LLM explanation..."
            )

            llm_result = generate_explanation(
                query=question,

                incident_date=(
                    parsed_date.isoformat()
                    if parsed_date
                    else None
                ),

                explanations=explanations,
            )

            print(
                "LLM trustworthy: "
                f"{llm_result.is_trustworthy}"
            )

        else:

            print(
                "\n[3/3] LLM generation skipped."
            )

        # ============================================================
        # SERIALIZABLE EXPLANATIONS
        # ============================================================

        serialized_explanations = []

        for exp in explanations:

            serialized_explanations.append(
                {
                    "law": exp.law,

                    "section": exp.section,

                    "heading": exp.heading,

                    "full_text": exp.full_text,

                    "law_reason": exp.law_reason,

                    "law_confidence": exp.law_confidence,

                    "law_warnings": exp.law_warnings,

                    "bm25_score_raw": exp.bm25_score_raw,

                    "bm25_score_norm": exp.bm25_score_norm,

                    "semantic_score_raw":
                        exp.semantic_score_raw,

                    "semantic_score_norm":
                        exp.semantic_score_norm,

                    "fused_score":
                        exp.fused_score,

                    "matched_keywords":
                        exp.matched_keywords,

                    "current_equivalent": {
                        "direction":
                            exp.current_equivalent.direction,

                        "status":
                            exp.current_equivalent.status,

                        "message":
                            exp.current_equivalent.message,

                        "sections":
                            exp.current_equivalent.sections,
                    },

                    "disclaimer":
                        exp.disclaimer,
                }
            )

        # ============================================================
        # SERIALIZE LLM RESULT
        # ============================================================

        serialized_llm_result = None

        if llm_result is not None:

            serialized_llm_result = {
                "query":
                    llm_result.query,

                "incident_date":
                    llm_result.incident_date,

                "response_text":
                    llm_result.response_text,

                "evidence_sections_used":
                    llm_result.evidence_sections_used,

                "uncited_section_flags":
                    llm_result.uncited_section_flags,

                "missing_citation":
                    llm_result.missing_citation,

                "trustworthy":
                    llm_result.is_trustworthy,
            }

        # ============================================================
        # FINAL RESULT
        # ============================================================

        result = {
            "question": question,

            "incident_date": (
                parsed_date.isoformat()
                if parsed_date
                else None
            ),

            "applicable_law":
                resolution.law,

            "law_reason":
                resolution.reason,

            "law_confidence":
                resolution.confidence,

            "law_warnings":
                list(resolution.warnings),

            "explanations":
                serialized_explanations,

            "llm_result":
                serialized_llm_result,
        }

        return result

    # =================================================================
    # SIMPLE ANSWER
    # =================================================================

    def answer(
        self,
        question: str,
        incident_date=None,
    ) -> str:
        """
        Convenience method.

        Returns only the grounded LLM answer.
        """

        result = self.query(
            question=question,

            incident_date=incident_date,

            generate_llm=True,
        )

        llm_result = result.get(
            "llm_result"
        )

        if llm_result is None:

            return (
                "No LLM response generated."
            )

        if not llm_result["trustworthy"]:

            return (
                "The generated answer failed "
                "the hallucination/citation checks "
                "and should not be relied upon."
            )

        return llm_result["response_text"]
    