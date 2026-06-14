"""
EXAMPLE: agents/coordinator/candidates.py
==========================================
PURPOSE: Package each (draft, critique) pair as a typed Candidate so the
         Coordinator can compare multiple revision rounds cleanly.

WHY THIS FILE EXISTS:
  After the Planner drafts an answer and Critics evaluates it, the Coordinator
  needs somewhere to store that pair. Without a Candidate model you end up
  passing raw strings between functions — which makes comparison, ranking, and
  iteration history hard to track. A typed container makes the data flow
  explicit and testable.

HOW TO USE THIS EXAMPLE:
  Read through, then try:
    1. Add a `timestamp` field to Candidate to track when it was created
    2. Write a function that picks the Candidate with the fewest issues
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Critique model (mirrors Critics agent output_schema from Phase 2)
# ---------------------------------------------------------------------------
# This is the structured output the Critics agent returns. Keeping it here
# (rather than importing from critics.py) makes candidates.py self-contained
# for learning purposes. In production, import it from a shared models file.

class Critique(BaseModel):
    issues: list[str] = Field(
        description="Concrete problems found in the draft — cite specific sentences"
    )
    missing_evidence: list[str] = Field(
        description="Claims that are made without a cited source"
    )
    suggested_revisions: list[str] = Field(
        description="Specific, actionable changes the Planner should make"
    )

    def is_clean(self, max_issues: int = 0) -> bool:
        """Return True if the critique has at most `max_issues` issues.

        LEARNING NOTE:
          evaluation.py calls this to decide whether to iterate or finalize.
          Setting max_issues=0 means the draft must be perfect before passing.
          Setting max_issues=1 means one minor issue is acceptable.
        """
        return len(self.issues) <= max_issues


# ---------------------------------------------------------------------------
# Candidate model
# ---------------------------------------------------------------------------
# A Candidate bundles one Planner draft with the Critics evaluation of it,
# plus metadata about which iteration produced it.

class Candidate(BaseModel):
    round: int = Field(description="Which iteration produced this candidate (1-indexed)")
    draft: str = Field(description="The Planner's answer text for this round")
    critique: Critique = Field(description="The Critics evaluation of this draft")

    def issue_count(self) -> int:
        """Total number of issues raised by Critics."""
        return len(self.critique.issues)

    def summary(self) -> str:
        """One-line description for logging and debugging."""
        return (
            f"Round {self.round}: {self.issue_count()} issue(s), "
            f"{len(self.critique.missing_evidence)} unsupported claim(s)"
        )


# ---------------------------------------------------------------------------
# Helper: pick best candidate from a list
# ---------------------------------------------------------------------------
# evaluation.py uses this when the iteration cap is reached and the
# Coordinator must pick the least-bad candidate from all rounds.

def best_candidate(candidates: list[Candidate]) -> Candidate:
    """Return the Candidate with the fewest issues.

    If two candidates tie on issue count, prefer the later round (more refined).

    LEARNING NOTE:
      This is intentionally simple. A smarter version might weight
      `missing_evidence` count more heavily than `issues`, or use an LLM
      to score candidates against the original query constraints.
    """
    return min(candidates, key=lambda c: (c.issue_count(), -c.round))


# ---------------------------------------------------------------------------
# HANDS-ON: Run this file directly and inspect the output
# ---------------------------------------------------------------------------
# Command: uv run python docs/candidates.py
#
# Try modifying best_candidate() to prefer later rounds when issue counts tie.

if __name__ == "__main__":
    round1 = Candidate(
        round=1,
        draft="RAG retrieves context at query time. Fine-tuning bakes knowledge into weights.",
        critique=Critique(
            issues=[
                "Does not mention cost trade-offs",
                "No discussion of inference latency",
            ],
            missing_evidence=["Inference speed claim has no source"],
            suggested_revisions=[
                "Add a section comparing update cost",
                "Cite a benchmark for latency",
            ],
        ),
    )

    round2 = Candidate(
        round=2,
        draft=(
            "RAG retrieves context at query time (cheaper to update); "
            "fine-tuning bakes knowledge into weights (faster inference). "
            "Choose RAG when your data changes frequently. [Source: 1]"
        ),
        critique=Critique(
            issues=["Could mention hybrid approaches"],
            missing_evidence=[],
            suggested_revisions=["Optional: add a note on RAG + fine-tuning combined"],
        ),
    )

    candidates = [round1, round2]
    for c in candidates:
        print(c.summary())

    best = best_candidate(candidates)
    print(f"\nBest candidate: {best.summary()}")
    print(f"Draft: {best.draft}")
