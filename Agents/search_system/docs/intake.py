"""
EXAMPLE: agents/coordinator/intake.py
======================================
PURPOSE: Parse and normalize the raw user query before anything is delegated.
         This runs first — all other coordinator steps consume its output.

WHY THIS FILE EXISTS:
  Without intake, every specialist receives the raw user message verbatim.
  Each agent may interpret phrasing, tone, or constraints differently, leading
  to inconsistent downstream results. Intake normalizes once so Websearcher,
  Planner, and Critics all work from the same structured input.

HOW TO USE THIS EXAMPLE:
  Read through the code and inline comments. Then try modifying:
    1. Add a new field to ParsedQuery (e.g. `language: str`)
    2. Write a new test case in the __main__ block below
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
# ParsedQuery is the structured output of intake. Every downstream step
# (Websearcher, Planner, Critics) receives this instead of the raw string.
# Using a Pydantic model means all fields are validated and typed —
# if `max_words` is accidentally set to "short" instead of an int, you'll
# catch it here rather than deep inside the Planner.

class ParsedQuery(BaseModel):
    question: str = Field(description="The core research question, rephrased for clarity")
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit constraints from the user, e.g. 'concise', 'bullet points'"
    )
    specialists_needed: list[str] = Field(
        default_factory=list,
        description="Which agents are likely needed: websearcher, planner, critics"
    )
    max_words: int | None = Field(
        default=None,
        description="Word limit parsed from the user's request, if any"
    )


# ---------------------------------------------------------------------------
# Intake function
# ---------------------------------------------------------------------------
# In production this would call an LlmAgent to extract structure from the
# raw message. Here we use simple heuristics so you can run it locally
# without an API key and see the shape of what intake produces.

def parse_query(raw: str) -> ParsedQuery:
    """Normalize a raw user message into a ParsedQuery.

    LEARNING NOTE:
      In a real coordinator, this function would call a small LlmAgent with
      a structured output schema (output_schema=ParsedQuery). The LLM handles
      ambiguous phrasing, implied constraints, and multi-part questions.
      The heuristic version below is enough to understand the data flow.

    Args:
        raw: The raw message from the user.

    Returns:
        A ParsedQuery with the core question, constraints, and routing hints.

    Example:
        >>> result = parse_query("Quickly explain RAG vs fine-tuning, keep it short")
        >>> result.question
        'What are the trade-offs between RAG and fine-tuning?'
        >>> "concise" in result.constraints
        True
    """
    question = raw.strip().rstrip("?") + "?"

    constraints = []
    if any(word in raw.lower() for word in ["quick", "short", "concise", "brief"]):
        constraints.append("concise")
    if "bullet" in raw.lower():
        constraints.append("bullet points")
    if "example" in raw.lower():
        constraints.append("include examples")

    # Estimate word limit from phrasing like "under 200 words"
    max_words = None
    import re
    match = re.search(r"under (\d+) words?", raw, re.IGNORECASE)
    if match:
        max_words = int(match.group(1))

    # All queries route to all three specialists in this system.
    # A more advanced intake might skip Websearcher for purely conceptual
    # questions or skip Critics for simple factual lookups.
    specialists_needed = ["websearcher", "planner", "critics"]

    return ParsedQuery(
        question=question,
        constraints=constraints,
        specialists_needed=specialists_needed,
        max_words=max_words,
    )


# ---------------------------------------------------------------------------
# HANDS-ON: Run this file directly and inspect the output
# ---------------------------------------------------------------------------
# Command: uv run python docs/intake.py
#
# Try changing the test inputs below to see how constraints are parsed.
# Then think about: what would you add to ParsedQuery for your own use case?

if __name__ == "__main__":
    test_cases = [
        "Quickly explain RAG vs fine-tuning, keep it short",
        "What are vector databases? Give me bullet points with examples",
        "Explain the A2A protocol under 150 words",
        "Compare LangChain and LlamaIndex",
    ]

    for raw in test_cases:
        result = parse_query(raw)
        print(f"\nInput:    {raw!r}")
        print(f"Question: {result.question}")
        print(f"Constraints: {result.constraints}")
        print(f"Max words:   {result.max_words}")
        print(f"Specialists: {result.specialists_needed}")
