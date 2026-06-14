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
from typing import List

class ParsedQuery(BaseModel):
    question: str = Field(description="The core research question, rephrased for clarity")
    constraints: List[str] = Field(
        default_factory=list,
        description="Explicit constraints from the user, e.g. 'concise', 'bullet points'"
    )
    specialists_needed: List[str] = Field(
        default_factory=list,
        description="Which agents are likely needed: websearcher, planner, critics"
    )
    max_words: int | None = Field(
        default = None,
        description="Word limit parsed from the user's request, if any",
    )

def parse_query(raw: str) -> ParsedQuery:
    """Normalizes a prompt into a structured key-value query for AI agents
    Args:
        raw(str): the raw string from users
    Returns:
        A PrasedQeury with the keys of the question, constraints, hints.
    """
    question = raw.strip().rstrip("?") + "?"
    constraints = []
    if any(word in raw.lower() for word in ["quick", "short", "concise", "brief"]):
        constraints.append("concise")
    if "bullet" in raw.lower():
        constraints.append("bullet points")
    if "example" in raw.lower():
        constraints.append("include example")
    
    max_words = None
    import re
    matched = re.search(r"under (\d+) words?", raw, re.IGNORECASE)
    if matched:
        max_words = int(matched.group(1))
    specialists_needed = ["websearcher", "planner", "critics"]
    return ParsedQuery (
        question=question,
        constraints=constraints,
        specialists_needed=specialists_needed,
        max_words=max_words,
    )

# Test case
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
