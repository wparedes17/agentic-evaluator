import re
from functools import lru_cache
from typing import Optional

from agent_patterns.patterns import ReflectionAgent

from app.config import Settings


EVALUATION_PROMPT_TEMPLATE = """You are an expert quality evaluator. Your task is to evaluate a response against a given guideline.

## Question
{question}

## Guideline
{guideline}

## Response to Evaluate
{response}

---

Provide a structured evaluation with the following sections:

### Assessment
A concise overall assessment of how well the response meets the guideline.

### Strengths
List specific strengths of the response relative to the guideline.

### Weaknesses
List specific weaknesses or gaps relative to the guideline.

### Score
Provide a score from 0 to 10 (decimals allowed) in this exact format:
SCORE: <number>

### Verdict
State PASS if the score is 7 or above, FAIL otherwise, in this exact format:
VERDICT: PASS or VERDICT: FAIL
"""


def _build_llm_configs(settings: Settings) -> dict:
    return {
        "documentation": {
            "provider": settings.llm_provider,
            "model_name": settings.llm_model_name,
            "temperature": settings.llm_temperature,
        },
        "reflection": {
            "provider": settings.llm_provider,
            "model_name": settings.llm_model_name,
            "temperature": 0.1,
        },
    }


@lru_cache(maxsize=1)
def _get_agent(provider: str, model_name: str, temperature: float, max_cycles: int) -> ReflectionAgent:
    """Create and cache a single ReflectionAgent instance."""
    settings_snapshot = Settings(
        llm_provider=provider,
        llm_model_name=model_name,
        llm_temperature=temperature,
        max_reflection_cycles=max_cycles,
    )
    return ReflectionAgent(
        llm_configs=_build_llm_configs(settings_snapshot),
        max_reflection_cycles=max_cycles,
    )


def _parse_score(text: str) -> Optional[float]:
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return max(0.0, min(10.0, value))
    return None


def _parse_verdict(text: str) -> Optional[bool]:
    match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "PASS"
    return None


def evaluate(
    question: str,
    guideline: str,
    response: str,
    settings: Settings,
) -> tuple[str, Optional[float], Optional[bool]]:
    """
    Evaluate a response against a guideline using a ReflectionAgent.

    Returns:
        (evaluation_text, score, passed)
    """
    agent = _get_agent(
        provider=settings.llm_provider,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_cycles=settings.max_reflection_cycles,
    )

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        guideline=guideline,
        response=response,
    )

    evaluation_text = agent.run(prompt)
    score = _parse_score(evaluation_text)
    passed = _parse_verdict(evaluation_text)

    # Derive passed from score if verdict is missing
    if passed is None and score is not None:
        passed = score >= 7.0

    return evaluation_text, score, passed
