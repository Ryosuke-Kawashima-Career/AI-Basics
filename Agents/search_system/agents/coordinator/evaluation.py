"""
EXAMPLE: agents/coordinator/evaluation.py
==========================================
PURPOSE: Decide after each Critics round whether to iterate (ask Planner to
         revise) or finalize (pass the best candidate to recommendation.py).
         Also enforces the iteration cap so the loop always terminates.
 
WHY THIS FILE EXISTS:
  Without evaluation logic, the Coordinator has no stopping rule. It would
  either loop forever (if Critics always finds something) or stop after exactly
  one round regardless of quality. evaluation.py makes that decision explicit
  and configurable — separate from the agents that produce and critique drafts.
 
HOW TO USE THIS EXAMPLE:
  Read through, then try:
    1. Change MAX_ROUNDS to 1 and see how the output changes
    2. Add a `severity` field to Critique and make evaluate() weight it
    3. Trace through the loop in __main__ manually before running it
"""
 
from __future__ import annotations
 
from candidates import Candidate, Critique, best_candidate
from typing import List
from pydantic import BaseModel, Field
from typing import Tuple, List

from candidates import Candidate, Critique, best_candidate

MAX_ROUNDS=3
MAX_ISSUES_TO_PASS=1

class EvaluationResult:
    def __init__(self, should_iterate: bool, reason: str, best: Candidate) -> None:
        self.should_iterate = should_iterate
        self.reason = reason
        self.best = best
      
    def __repr__(self) -> str:
        action = "ITERATE" if self.should_iterate else "FINALIZE"
        return f"EvaluationResult({action}: {self.reason})"

def evaluate(candidates: List[Candidate]) -> EvaluationResult:
    """Decide whether to iterate or finalize based on all rounds so far.
    Called by the coordinator after each Critics's response. This shows what the Coordinator has to do.
    """
    if len(candidates) == 0:
        raise ValueError("evaluate() called with no candidates")
    latest: Candidate = candidates[-1]
    best: Candidate = best_candidate(candidates)
    if latest.round >= MAX_ROUNDS:
        return EvaluationResult(
            should_iterate=False,
            reason=f"Reached the max rounds of {MAX_ROUNDS}",
            best=best,
        )
    if latest.critique.is_clean(max_issues=MAX_ISSUES_TO_PASS):
        return EvaluationResult(
            should_iterate=False,
            reason=(
                "Draft is good enough" + 
                f"({latest.issue_count()}) issue(s) <= threshold {MAX_ISSUES_TO_PASS}"
            ),
            best=latest,
        )
    return EvaluationResult(
        should_iterate=True,
        reason=(
            f"Round {latest.round}: {latest.issue_count()} issue(s) found, "
            f"asking Planner to revise"
        ),
        best=latest,
    )
if __name__ == "__main__":
    # Simulate three rounds of improving drafts
    simulated_rounds = [
        Candidate(
            round=1,
            draft="RAG retrieves context at query time.",
            critique=Critique(
                issues=["Missing cost comparison", "No latency discussion"],
                missing_evidence=["No source for inference speed"],
                suggested_revisions=["Add cost section", "Cite a latency benchmark"],
            ),
        ),
        Candidate(
            round=2,
            draft="RAG is cheaper to update; fine-tuning is faster at inference. [Source: 1]",
            critique=Critique(
                issues=["Could mention hybrid approaches"],
                missing_evidence=[],
                suggested_revisions=["Optional: note on RAG + fine-tuning combined"],
            ),
        ),
        Candidate(
            round=3,
            draft=(
                "RAG retrieves context at query time (cheaper to update). "
                "Fine-tuning bakes knowledge into weights (faster inference). "
                "Hybrid approaches combine both. [Sources: 1, 2]"
            ),
            critique=Critique(
                issues=[],
                missing_evidence=[],
                suggested_revisions=[],
            ),
        ),
    ]
 
    # Replay the evaluation loop round by round
    accumulated: list[Candidate] = []
    for candidate in simulated_rounds:
        accumulated.append(candidate)
        result = evaluate(accumulated)
        print(f"\nAfter round {candidate.round}: {result}")
        if not result.should_iterate:
            print(f"\n→ Final answer selected from: {result.best.summary()}")
            print(f"  Draft: {result.best.draft}")
            break
