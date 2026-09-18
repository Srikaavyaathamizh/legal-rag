import sys

# Allow imports from src/
sys.path.insert(0, "src")

from llm_explainer import generate_explanation


class FakeExplanation:
    def __init__(self, law, section, heading, full_text):
        self.law = law
        self.section = section
        self.heading = heading
        self.full_text = full_text


evidence = [
    FakeExplanation(
        "IPC",
        "304A",
        "Causing death by negligence",
        "Whoever causes the death of any person by doing "
        "any rash or negligent act not amounting to culpable "
        "homicide shall be punished with imprisonment of either "
        "description for a term which may extend to two years, "
        "or with fine, or with both."
    )
]


result = generate_explanation(
    query="What happens if someone causes death by negligent driving?",
    incident_date="2023-12-10",
    explanations=evidence,
)


print("=" * 70)
print("--- LLM RESPONSE ---")
print(result.response_text)

print("=" * 70)
print("--- Evidence Sections Used ---")
print(result.evidence_sections_used)

print("=" * 70)
print("--- Uncited Section Flags ---")
print(result.uncited_section_flags)

print("=" * 70)
print("--- Missing Citation? ---")
print(result.missing_citation)

print("=" * 70)
print("--- Trustworthy? ---")
print(result.is_trustworthy)

print("=" * 70)