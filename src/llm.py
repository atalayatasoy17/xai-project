"""LLM generation utilities for evidence-grounded explanations."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_FORBIDDEN_PHRASES = [
    "true label",
    "correct prediction",
    "incorrect prediction",
    "consistent with the true outcome",
    "inconsistent with the true outcome",
    "stable",
    "stability",
    "adequate",
    "protective",
    "normal",
    "abnormal",
    "moderate",
]


GENERATOR_SYSTEM_MESSAGE = """
You are a clinical AI explanation generator.

Your role is to generate a concise, evidence-grounded explanation for an ICU mortality prediction model.

Hard rules:
- Use only the provided evidence.
- Do not invent clinical facts.
- Do not mention or use the true label in the explanation.
- Do not say the prediction was correct, incorrect, consistent with the true outcome, or inconsistent with the true outcome.
- Do not add measurement units unless they are explicitly present in the evidence.
- Do not invent normal ranges, diagnoses, mechanisms, or clinical details beyond the provided clinical_meaning.
- If clinical_meaning is "No predefined clinical interpretation available.", do not infer a clinical interpretation for that feature.
- For features without explicit clinical_meaning, only state that they increased or decreased the model's predicted risk.
- For features without explicit clinical_meaning, do not describe the value as low, high, normal, adequate, stable, unstable, elevated, reduced, or moderate.
- For features with "No predefined clinical interpretation available.", use this exact style: "<feature> = <value> increased/decreased the model's predicted risk."
- Do not paraphrase feature names when clinical_meaning is not available.
- Do not use interpretive adjectives such as protective, adequate, stable, unstable, normal, abnormal, elevated, low, high, or moderate unless the provided clinical_meaning explicitly supports that wording.
- Correctly distinguish risk-increasing and risk-decreasing evidence.
- Mention caution flags when present.
- If a value has a caution flag, explain that it may require careful interpretation.
- Keep the explanation clinically grounded and concise.

Write the explanation using these sections:
1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation
""".strip()


def load_openai_client() -> OpenAI:
    """Load the OpenAI client from OPENAI_API_KEY in the environment or .env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file or environment."
        )

    return OpenAI(api_key=api_key)


def generate_explanation(
    prompt: str,
    client: OpenAI | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
) -> str:
    """Generate an evidence-grounded explanation from a prepared prompt."""
    if client is None:
        client = load_openai_client()

    response = client.responses.create(
        model=model,
        instructions=GENERATOR_SYSTEM_MESSAGE,
        input=prompt,
        temperature=temperature,
    )

    return response.output_text


def check_forbidden_phrases(
    text: str,
    forbidden_phrases: list[str] | None = None,
) -> list[str]:
    """Return forbidden or cautionary phrases found in generated text."""
    phrases = forbidden_phrases or DEFAULT_FORBIDDEN_PHRASES
    lower_text = text.lower()

    return [phrase for phrase in phrases if phrase.lower() in lower_text]
