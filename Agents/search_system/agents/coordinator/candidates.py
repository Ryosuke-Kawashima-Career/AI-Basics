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
from typing import List, Tuple

class Critique(BaseModel):
    issues: List[str] = Field(
        description="Concrete problems found in the draft - cite specific sentences"
    )
    missing_evidence: List[str] = Field(
        description="Claims that are made without a cited source"
    )
    suggested_revisions: List[str] = Field(
        description="Specific, actionable changes the planner should make"
    )

    def is_clean(self, max_issues: int = 0) -> bool:
        """Return the judgement whether the output is good or not"""
        return len(self.issues) <= max_issues

class Candidate(BaseModel):
    round:int = Field(description="which iteration produced this candidate")
    draft: str = Field(description="The planner's answer for this round")
    critique: Critique = Field(description="The critics evaluation of this draft")

    def issue_count(self) -> int:
        """Returns the number of issues"""
        return len(self.critique.issues)

    def summary(self) -> Tuple[str, str]:
        """One-line description for logging and debugging"""
        return (
            f"Round {self.round}: {self.issue_count()} issues",
            f"{len(self.critique.missing_evidence)} unsupported claims"
        )
    
def best_candidate(candidates: List[Candidate]) -> Candidate:
    """Return the candidate with the fewest issues"""
    return min(candidates, key=lambda c: (c.issue_count(), -c.round))   
 
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