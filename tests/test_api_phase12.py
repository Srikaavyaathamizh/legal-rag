"""
Phase 12 — Legal RAG API Test Suite
------------------------------------

Tests:
1. Health endpoint
2. Pre-BNS date -> IPC
3. Post-BNS date -> BNS
4. Missing date -> AMBIGUOUS
5. Invalid date
6. Empty question
7. IPC -> BNS equivalent
8. BNS -> IPC historical equivalent
9. Important section retrieval
"""

import requests


BASE_URL = "http://127.0.0.1:8000"


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# TEST 1 — Health
# =========================================================

def test_health():

    print_header("TEST 1 — Health endpoint")

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=30,
    )

    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    print("PASS")


# =========================================================
# TEST 2 — Pre-BNS date
# =========================================================

def test_ipc_query():

    print_header("TEST 2 — Pre-BNS date -> IPC")

    payload = {
        "question": "Someone died because of negligent driving",
        "incident_date": "2023-12-10",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    print("Status:", response.status_code)

    assert response.status_code == 200

    data = response.json()

    print("Applicable law:", data["applicable_law"])
    print("Answer:", data["answer"])

    assert data["applicable_law"] == "IPC"

    assert data["incident_date"] == "2023-12-10"

    assert data["trustworthy"] is True

    sections = [
        item["section"]
        for item in data["retrieved_sections"]
    ]

    print("Retrieved:", sections)

    assert "304A" in sections

    print("PASS — IPC correctly selected")


# =========================================================
# TEST 3 — Post-BNS date
# =========================================================

def test_bns_query():

    print_header("TEST 3 — Post-BNS date -> BNS")

    payload = {
        "question": "What provision covers causing death by negligence?",
        "incident_date": "2025-01-01",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    print("Status:", response.status_code)

    assert response.status_code == 200

    data = response.json()

    print("Applicable law:", data["applicable_law"])
    print("Answer:", data["answer"])

    assert data["applicable_law"] == "BNS"

    sections = [
        item["section"]
        for item in data["retrieved_sections"]
    ]

    print("Retrieved:", sections)

    assert "106" in sections

    print("PASS — BNS correctly selected")


# =========================================================
# TEST 4 — Missing date
# =========================================================

def test_ambiguous_date():

    print_header("TEST 4 — Missing date -> AMBIGUOUS")

    payload = {
        "question": "What provision covers causing death by negligence?"
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    print("Status:", response.status_code)

    assert response.status_code == 200

    data = response.json()

    print("Applicable law:", data["applicable_law"])
    print("Warning:", data["law_warnings"])

    assert data["applicable_law"] == "AMBIGUOUS"

    assert data["law_confidence"] == "low"

    assert len(data["law_warnings"]) > 0

    print("PASS — ambiguous-date handling works")


# =========================================================
# TEST 5 — Invalid date
# =========================================================

def test_invalid_date():

    print_header("TEST 5 — Invalid date")

    payload = {
        "question": "What provision covers murder?",
        "incident_date": "2025-99-99",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=30,
    )

    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 400

    detail = response.json()["detail"]

    assert "Invalid incident_date" in detail

    print("PASS — invalid date rejected")


# =========================================================
# TEST 6 — Empty question
# =========================================================

def test_empty_question():

    print_header("TEST 6 — Empty question")

    payload = {
        "question": "   ",
        "incident_date": "2025-01-01",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=30,
    )

    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 400

    assert "Question cannot be empty" in response.json()["detail"]

    print("PASS — empty question rejected")


# =========================================================
# TEST 7 — IPC -> BNS equivalent
# =========================================================

def test_ipc_current_equivalent():

    print_header("TEST 7 — IPC -> BNS equivalent")

    payload = {
        "question": "Someone died because of negligence",
        "incident_date": "2023-12-10",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["applicable_law"] == "IPC"

    found = False

    for section in data["retrieved_sections"]:

        if section["section"] == "304A":

            print(
                "304A equivalent:",
                section["current_equivalent"],
            )

            assert "BNS Section 106" in section["current_equivalent"]

            found = True

    assert found

    print("PASS — IPC 304A maps to BNS 106")


# =========================================================
# TEST 8 — BNS -> IPC historical equivalent
# =========================================================

def test_bns_historical_equivalent():

    print_header("TEST 8 — BNS -> IPC historical equivalent")

    payload = {
        "question": "What provision covers causing death by negligence?",
        "incident_date": "2025-01-01",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["applicable_law"] == "BNS"

    found = False

    for section in data["retrieved_sections"]:

        if section["section"] == "106":

            print(
                "BNS 106 equivalent:",
                section["current_equivalent"],
            )

            assert "IPC Section 304A" in section["current_equivalent"]

            found = True

    assert found

    print("PASS — BNS 106 maps to IPC 304A")


# =========================================================
# TEST 9 — Citation / grounding
# =========================================================

def test_grounding():

    print_header("TEST 9 — Grounded answer / citations")

    payload = {
        "question": "Someone died because of negligent driving",
        "incident_date": "2023-12-10",
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    assert response.status_code == 200

    data = response.json()

    print("Trustworthy:", data["trustworthy"])
    print("Evidence:", data["evidence_sections"])
    print("Uncited:", data["uncited_sections"])
    print("Missing citation:", data["missing_citation"])

    assert data["trustworthy"] is True

    assert len(data["evidence_sections"]) > 0

    assert data["missing_citation"] is False

    assert data["answer"]

    print("PASS — grounded answer and citation checks work")


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PHASE 12 — LEGAL RAG API TEST SUITE")
    print("=" * 70)

    test_health()

    test_ipc_query()

    test_bns_query()

    test_ambiguous_date()

    test_invalid_date()

    test_empty_question()

    test_ipc_current_equivalent()

    test_bns_historical_equivalent()

    test_grounding()

    print("\n" + "=" * 70)
    print("PHASE 12 API TEST SUITE COMPLETED")
    print("=" * 70)