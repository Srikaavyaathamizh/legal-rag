"""
Phase 7 — LLM Explanation + Hallucination Guard
--------------------------------------------------
Turns retrieved, explained evidence (Phase 6) into a plain-language
answer with an explicit citation.

The LLM is deliberately restricted:

  - It only ever sees the retrieved section text.
  - It is instructed to say "insufficient evidence" rather than guess.
  - It is instructed never to invent section numbers.

Defense in depth:
`detect_uncited_sections()` checks the generated answer for section
numbers that were not present in the supplied evidence.

Requires:
    pip install groq

and:

    GROQ_API_KEY

in the environment.
"""

from __future__ import annotations

import os
import re

from dataclasses import dataclass, field
from typing import Optional


SYSTEM_PROMPT = """You are a legal information assistant for Indian criminal law (IPC and BNS).

STRICT RULES — follow these exactly:

1. Answer ONLY using the legal provisions given to you in the "EVIDENCE" section below. Do not use any other knowledge of Indian law you may have.

2. If the evidence provided is empty, or does not actually address the question, say explicitly:
"I do not have sufficient evidence to answer this confidently."
Do not attempt to answer from general knowledge.

3. Never state or imply a section number that does not appear verbatim in the evidence provided.

4. Do not invent, guess, or extrapolate legal provisions, penalties, or procedures not present in the evidence.

5. Write a short, plain-language explanation (2-4 sentences) of what the top matching provision means in practice.

6. End your answer with an explicit citation line in exactly this format:
"Source: <LAW> Section <SECTION>"

Example:
Source: IPC Section 304A

If multiple provisions were used, cite each on its own line.

7. Do not give legal advice, predictions about case outcomes, or procedural instructions beyond what the evidence states.
"""


def build_evidence_block(
    explanations: list,
    max_items: int = 3
) -> str:
    """
    Renders only the retrieved section text as evidence.
    """

    if not explanations:
        return "(No relevant legal provisions were retrieved for this query.)"

    blocks = []

    for exp in explanations[:max_items]:
        blocks.append(
            f"[{exp.law} Section {exp.section}] {exp.heading}\n"
            f"{exp.full_text}"
        )

    return "\n\n".join(blocks)


def build_user_prompt(
    query: str,
    incident_date: Optional[str],
    explanations: list,
    max_items: int = 3
) -> str:
    """
    Build the user prompt containing the question,
    incident date, and retrieved evidence.
    """

    date_line = (
        f"Incident date: {incident_date}"
        if incident_date
        else "Incident date: not provided"
    )

    evidence = build_evidence_block(
        explanations,
        max_items=max_items
    )

    return (
        f"QUESTION: {query}\n"
        f"{date_line}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"Answer the question using only the evidence above, "
        f"following all rules in the system prompt."
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "openai/gpt-oss-120b",
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> str:
    """
    Real API call through Groq.

    Requires GROQ_API_KEY in the environment.
    """

    # Import only when the real LLM call is made.
    from groq import Groq

    # IMPORTANT:
    # Read the actual GROQ_API_KEY environment variable.
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com/keys "
            "and set it in your environment before calling call_llm()."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
    )

    return response.choices[0].message.content


SECTION_MENTION_RE = re.compile(
    r"\bSection\s+(\d+[A-Za-z]{0,2}(?:\(\d+\))?)",
    re.IGNORECASE
)


def detect_uncited_sections(
    response_text: str,
    evidence_sections: list[str]
) -> list[str]:
    """
    Extract every "Section N" mention from the response and flag
    section numbers that were not present in the evidence.
    """

    mentioned = {
        m.group(1)
        for m in SECTION_MENTION_RE.finditer(response_text)
    }

    evidence_set = set(evidence_sections)

    return sorted(
        mentioned - evidence_set
    )


def detect_missing_citation(
    response_text: str
) -> bool:
    """
    True if the response does not contain a Source: citation.
    """

    return "Source:" not in response_text


@dataclass
class LLMExplanationResult:
    query: str
    incident_date: Optional[str]
    response_text: str
    evidence_sections_used: list[str]
    uncited_section_flags: list[str] = field(
        default_factory=list
    )
    missing_citation: bool = False

    @property
    def is_trustworthy(self) -> bool:
        """
        Conservative trustworthiness gate.

        The response is considered trustworthy only if:

        1. No section hallucination was detected.
        2. A Source: citation exists.
        """

        return (
            not self.uncited_section_flags
            and not self.missing_citation
        )


def generate_explanation(
    query: str,
    incident_date: Optional[str],
    explanations: list,
    llm_fn=call_llm,
    max_items: int = 3,
) -> LLMExplanationResult:
    """
    Full Phase 7 pipeline:

        Evidence
            ↓
        Prompt construction
            ↓
        LLM
            ↓
        Hallucination detection
            ↓
        Trustworthiness result

    llm_fn is injectable so tests can use a mock LLM.
    """

    user_prompt = build_user_prompt(
        query,
        incident_date,
        explanations,
        max_items=max_items
    )

    response_text = llm_fn(
        SYSTEM_PROMPT,
        user_prompt
    )

    evidence_sections = [
        exp.section
        for exp in explanations[:max_items]
    ]

    uncited = detect_uncited_sections(
        response_text,
        evidence_sections
    )

    missing_citation = (
        detect_missing_citation(response_text)
        and bool(explanations)
    )

    return LLMExplanationResult(
        query=query,
        incident_date=incident_date,
        response_text=response_text,
        evidence_sections_used=evidence_sections,
        uncited_section_flags=uncited,
        missing_citation=missing_citation,
    )