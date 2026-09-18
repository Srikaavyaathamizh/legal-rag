"""
Phase 11 — Legal RAG API
------------------------

Exposes the Legal RAG pipeline through HTTP.

Endpoints:
    GET  /health
    POST /query
"""

from __future__ import annotations


import os
import sys
from datetime import date
from typing import Optional



# ---------------------------------------------------------
# Allow imports from ../src
# ---------------------------------------------------------

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pipeline import LegalRAGPipeline


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Legal RAG API",
    description=(
        "Temporal-aware Indian criminal law retrieval using "
        "IPC/BNS filtering, BM25, semantic search, hybrid ranking, "
        "explainability and grounded LLM generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    incident_date: Optional[str] = None


# ---------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------

print("=" * 70)
print("STARTING LEGAL RAG API")
print("=" * 70)

pipeline = LegalRAGPipeline(
    alpha=0.3,
    retrieval_k=30,
    top_k=3,
)

print("=" * 70)
print("LEGAL RAG API READY")
print("=" * 70)


# ---------------------------------------------------------
# Helper: Explanation -> dictionary
# ---------------------------------------------------------

def explanation_to_dict(exp):
    """
    Convert an Explanation dataclass/object OR dictionary
    into a JSON-serializable dictionary.
    """

    # -----------------------------------------------------
    # Case 1 — dictionary
    # -----------------------------------------------------

    if isinstance(exp, dict):

        current_equivalent = exp.get("current_equivalent")

        if isinstance(current_equivalent, dict):
            current_equivalent = current_equivalent.get(
                "message",
                ""
            )

        return {
            "law": exp.get("law"),

            "section": exp.get("section"),

            "heading": exp.get("heading"),

            "full_text": exp.get(
                "full_text",
                exp.get("text", "")
            ),

            "bm25_score_raw": exp.get(
                "bm25_score_raw",
                exp.get("bm25_score")
            ),

            "bm25_score_norm": exp.get(
                "bm25_score_norm"
            ),

            "semantic_score_raw": exp.get(
                "semantic_score_raw",
                exp.get("semantic_score")
            ),

            "semantic_score_norm": exp.get(
                "semantic_score_norm"
            ),

            "fused_score": exp.get(
                "fused_score"
            ),

            "matched_keywords": exp.get(
                "matched_keywords",
                []
            ),

            "current_equivalent": current_equivalent,
        }

    # -----------------------------------------------------
    # Case 2 — Explanation dataclass/object
    # -----------------------------------------------------

    current_equivalent = getattr(
        exp,
        "current_equivalent",
        None
    )

    if current_equivalent is not None:

        current_equivalent = getattr(
            current_equivalent,
            "message",
            str(current_equivalent)
        )

    return {
        "law": getattr(exp, "law", None),

        "section": getattr(exp, "section", None),

        "heading": getattr(exp, "heading", None),

        "full_text": getattr(exp, "full_text", ""),

        "bm25_score_raw": getattr(
            exp,
            "bm25_score_raw",
            None
        ),

        "bm25_score_norm": getattr(
            exp,
            "bm25_score_norm",
            None
        ),

        "semantic_score_raw": getattr(
            exp,
            "semantic_score_raw",
            None
        ),

        "semantic_score_norm": getattr(
            exp,
            "semantic_score_norm",
            None
        ),

        "fused_score": getattr(
            exp,
            "fused_score",
            None
        ),

        "matched_keywords": getattr(
            exp,
            "matched_keywords",
            []
        ),

        "current_equivalent": current_equivalent,
    }


# ---------------------------------------------------------
# Helper: LLM result extraction
# ---------------------------------------------------------

def llm_result_to_dict(llm_result):
    """
    Safely extract the LLMExplanationResult.

    Supports both the dataclass/object version and
    dictionary version.
    """

    if llm_result is None:
        return {
            "answer": None,
            "trustworthy": False,
            "evidence_sections": [],
            "uncited_sections": [],
            "missing_citation": True,
        }

    # -----------------------------------------------------
    # Dictionary
    # -----------------------------------------------------

    if isinstance(llm_result, dict):

        return {
            "answer": llm_result.get(
                "response_text",
                llm_result.get("answer", "")
            ),

            "trustworthy": llm_result.get(
                "is_trustworthy",
                llm_result.get("trustworthy", False)
            ),

            "evidence_sections": llm_result.get(
                "evidence_sections_used",
                llm_result.get("evidence_sections", [])
            ),

            "uncited_sections": llm_result.get(
                "uncited_section_flags",
                llm_result.get("uncited_sections", [])
            ),

            "missing_citation": llm_result.get(
                "missing_citation",
                True
            ),
        }

    # -----------------------------------------------------
    # Dataclass/object
    # -----------------------------------------------------

    return {
        "answer": getattr(
            llm_result,
            "response_text",
            ""
        ),

        "trustworthy": getattr(
            llm_result,
            "is_trustworthy",
            False
        ),

        "evidence_sections": getattr(
            llm_result,
            "evidence_sections_used",
            []
        ),

        "uncited_sections": getattr(
            llm_result,
            "uncited_section_flags",
            []
        ),

        "missing_citation": getattr(
            llm_result,
            "missing_citation",
            True
        ),
    }


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Legal RAG API",
    }


# ---------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # -----------------------------------------------------
    # Validate date
    # -----------------------------------------------------

    parsed_date = None

    if request.incident_date:

        try:

            parsed_date = date.fromisoformat(
                request.incident_date
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid incident_date. "
                    "Use YYYY-MM-DD format."
                )
            )

    # -----------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------

    try:

        result = pipeline.query(
            question=request.question,
            incident_date=parsed_date,
            generate_llm=True,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        print("PIPELINE ERROR:")
        print(repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # -----------------------------------------------------
    # Extract LLM result
    # -----------------------------------------------------

    llm_result = result.get(
        "llm_result"
    )

    llm_data = llm_result_to_dict(
        llm_result
    )

    # -----------------------------------------------------
    # Extract explanations
    # -----------------------------------------------------

    explanations = result.get(
        "explanations",
        []
    )

    retrieved_sections = []

    for exp in explanations:

        retrieved_sections.append(
            explanation_to_dict(exp)
        )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {

        "question": result.get(
            "question"
        ),

        "incident_date": result.get(
            "incident_date"
        ),

        "applicable_law": result.get(
            "applicable_law"
        ),

        "law_reason": result.get(
            "law_reason"
        ),

        "law_confidence": result.get(
            "law_confidence"
        ),

        "law_warnings": result.get(
            "law_warnings",
            []
        ),

        # ---------------------------------------------
        # LLM
        # ---------------------------------------------

        "answer": llm_data["answer"],

        "trustworthy": llm_data["trustworthy"],

        "evidence_sections": llm_data[
            "evidence_sections"
        ],

        "uncited_sections": llm_data[
            "uncited_sections"
        ],

        "missing_citation": llm_data[
            "missing_citation"
        ],

        # ---------------------------------------------
        # Retrieval
        # ---------------------------------------------

        "retrieved_sections": retrieved_sections,
    }