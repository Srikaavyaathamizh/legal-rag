import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_explainer import (
    build_user_prompt,
    detect_uncited_sections,
    detect_missing_citation,
    generate_explanation,
    SYSTEM_PROMPT,
)


# --- lightweight stand-in for a real Explanation object (Phase 6) ---
class FakeExplanation:
    def __init__(self, law, section, heading, full_text):
        self.law = law
        self.section = section
        self.heading = heading
        self.full_text = full_text


evidence_304a = FakeExplanation(
    "IPC", "304A", "Causing death by negligence",
    "Whoever causes the death of any person by doing any rash or negligent "
    "act not amounting to culpable homicide shall be punished with imprisonment "
    "of either description for a term which may extend to two years, or with "
    "fine, or with both.",
)
evidence_420 = FakeExplanation(
    "IPC", "420", "Cheating and dishonestly inducing delivery of property",
    "Whoever cheats and thereby dishonestly induces the person deceived to "
    "deliver any property to any person... shall be punished with imprisonment "
    "of either description for a term which may extend to seven years.",
)


print("=" * 70)
print("TEST 1 — Prompt construction includes only the retrieved evidence text")
prompt = build_user_prompt("What happens if someone causes death by negligence?", "2023-12-10", [evidence_304a])
print(prompt)
assert "304A" in prompt
assert "rash or negligent act" in prompt
assert "420" not in prompt  # section not in evidence must not leak into prompt
print("PASS — only 304A appears, 420 correctly absent")

print("=" * 70)
print("TEST 2 — Empty evidence produces an explicit 'no evidence' prompt, not a blank one")
prompt_empty = build_user_prompt("Some obscure question", "2023-01-01", [])
print(prompt_empty)
assert "No relevant legal provisions were retrieved" in prompt_empty
print("PASS")

print("=" * 70)
print("TEST 3 — Mock LLM: well-behaved response (correct citation, no hallucination) -> is_trustworthy=True")
def mock_llm_good(system_prompt, user_prompt):
    return (
        "This provision covers deaths caused by a rash or negligent act that does not "
        "amount to culpable homicide, such as fatal negligent driving.\n\n"
        "Source: IPC Section 304A"
    )
result = generate_explanation("death by negligent driving", "2023-12-10", [evidence_304a], llm_fn=mock_llm_good)
print(result)
assert result.is_trustworthy
assert result.uncited_section_flags == []
assert not result.missing_citation
print("PASS")

print("=" * 70)
print("TEST 4 — Mock LLM: HALLUCINATES a section never in evidence -> caught by post-hoc check")
def mock_llm_hallucinating(system_prompt, user_prompt):
    return (
        "This is covered under IPC Section 304A, and related provisions include "
        "Section 279 for rash driving and Section 337 for causing hurt.\n\n"
        "Source: IPC Section 304A"
    )
result = generate_explanation("death by negligent driving", "2023-12-10", [evidence_304a], llm_fn=mock_llm_hallucinating)
print(result)
assert not result.is_trustworthy, "should have been flagged untrustworthy"
assert "279" in result.uncited_section_flags
assert "337" in result.uncited_section_flags
assert "304A" not in result.uncited_section_flags  # this one WAS in evidence, must not be falsely flagged
print("PASS — correctly flags 279 and 337 as uncited, correctly does NOT flag 304A")

print("=" * 70)
print("TEST 5 — Mock LLM: forgets the citation line entirely -> caught")
def mock_llm_no_citation(system_prompt, user_prompt):
    return "This covers negligent acts causing death, punishable by up to two years imprisonment."
result = generate_explanation("death by negligent driving", "2023-12-10", [evidence_304a], llm_fn=mock_llm_no_citation)
assert result.missing_citation is True
assert not result.is_trustworthy
print("PASS — missing citation correctly caught")

print("=" * 70)
print("TEST 6 — Mock LLM given NO evidence correctly says so; missing_citation should NOT fire (no evidence = no expectation of a cite)")
def mock_llm_insufficient(system_prompt, user_prompt):
    return "I do not have sufficient evidence to answer this confidently."
result = generate_explanation("some completely unrelated question", "2023-01-01", [], llm_fn=mock_llm_insufficient)
print(result)
assert result.missing_citation is False, "should not penalize missing citation when there was no evidence to cite"
assert result.is_trustworthy
print("PASS")

print("=" * 70)
print("TEST 7 — detect_uncited_sections is law-agnostic on number only (documented limitation), verify as designed")
# IPC 302 and BNS 103 both exist; check the function only compares bare numbers
flags = detect_uncited_sections("See Section 302 and Section 999.", evidence_sections=["302"])
assert flags == ["999"]
print("PASS — 302 correctly not flagged (was in evidence), 999 correctly flagged")

print("=" * 70)
print("TEST 8 — max_items truncation: only top-3 evidence items reach the prompt even if more are passed")
many_evidence = [evidence_304a, evidence_420, evidence_304a, evidence_420, evidence_304a]
prompt = build_user_prompt("test", "2023-01-01", many_evidence, max_items=3)
occurrences = prompt.count("[IPC Section")
assert occurrences == 3, f"expected 3 evidence blocks, got {occurrences}"
print("PASS — evidence correctly capped at max_items")
