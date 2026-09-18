import sys
import os
import pytest

# Allow Python to import modules from ../src
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from pipeline import LegalRAGPipeline


print("=" * 70)
print("PHASE 10 — PIPELINE TEST")
print("=" * 70)


# ------------------------------------------------------------------
# SHARED PIPELINE
# ------------------------------------------------------------------
#
# IMPORTANT:
# Qdrant local persistent mode allows only one client to own the
# qdrant_data directory at a time.
#
# Therefore we create ONE pipeline and reuse it for all tests.
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():

    print("\n" + "=" * 70)
    print("CREATING SHARED LEGAL RAG PIPELINE")
    print("=" * 70)

    instance = LegalRAGPipeline(
        alpha=0.3,
        retrieval_k=30,
        top_k=5,
    )

    yield instance

    # --------------------------------------------------------------
    # Close Qdrant cleanly after all tests
    # --------------------------------------------------------------

    try:
        if hasattr(instance, "semantic_index"):
            if hasattr(instance.semantic_index, "client"):
                instance.semantic_index.client.close()
    except Exception as exc:
        print(f"Warning while closing Qdrant: {exc}")


# ------------------------------------------------------------------
# TEST 1 — IPC
# ------------------------------------------------------------------

def test_ipc_pipeline(pipeline):

    print("=" * 70)
    print("PHASE 10 — IPC PIPELINE TEST")
    print("=" * 70)

    result = pipeline.query(
        question="Someone died because of negligent driving",
        incident_date="2023-12-10",
    )

    print("\nApplicable law:")
    print(result["applicable_law"])

    print("\nFinal answer:")
    print(result["llm_result"]["response_text"])

    print("\nTrustworthy:")
    print(result["llm_result"]["trustworthy"])

    assert result["applicable_law"] == "IPC"

    assert result["llm_result"]["trustworthy"]

    assert "304A" in result["llm_result"]["response_text"]

    print("\nPASS — IPC pipeline test")


# ------------------------------------------------------------------
# TEST 2 — BNS
# ------------------------------------------------------------------

def test_bns_pipeline(pipeline):

    print("=" * 70)
    print("PHASE 10 — BNS PIPELINE TEST")
    print("=" * 70)

    result = pipeline.query(
        question="What provision covers causing death by negligence?",
        incident_date="2025-01-01",
    )

    print("\nApplicable law:")
    print(result["applicable_law"])

    print("\nFinal answer:")
    print(result["llm_result"]["response_text"])

    assert result["applicable_law"] == "BNS"

    print("\nPASS — BNS pipeline test")


# ------------------------------------------------------------------
# TEST 3 — AMBIGUOUS DATE
# ------------------------------------------------------------------

def test_ambiguous_date_pipeline(pipeline):

    print("=" * 70)
    print("PHASE 10 — AMBIGUOUS DATE PIPELINE TEST")
    print("=" * 70)

    result = pipeline.query(
        question="What provision covers causing death by negligence?",
        incident_date=None,
    )

    print("\nApplicable law:")
    print(result["applicable_law"])

    print("\nWarnings:")

    for warning in result["law_warnings"]:
        print("-", warning)

    assert result["applicable_law"] == "AMBIGUOUS"

    print("\nPASS — ambiguous-date pipeline test")


# ------------------------------------------------------------------
# END
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 10 PIPELINE TESTS READY")
print("=" * 70)
