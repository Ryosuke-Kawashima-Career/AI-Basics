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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keep these as module-level constants so they're easy to tune without
# touching the logic. In production these could come from an env var or
# a config file loaded at startup.

MAX_ROUNDS = 3          # Hard cap on Planner → Critics iterations
MAX_ISSUES_TO_PASS = 1  # A critique with <= this many issues is "good enough"


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------
# The output of evaluate() — tells the Coordinator what to do next.

class EvaluationResult:
    def __init__(self, should_iterate: bool, reason: str, best: Candidate):
        self.should_iterate = should_iterate  # True → ask Planner to revise
        self.reason = reason                  # Human-readable explanation
        self.best = best                      # The best candidate so far

    def __repr__(self) -> str:
        action = "ITERATE" if self.should_iterate else "FINALIZE"
        return f"EvaluationResult({action}: {self.reason})"


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(candidates: list[Candidate]) -> EvaluationResult:
    """Decide whether to iterate or finalize based on all rounds so far.

    Called by the Coordinator after each Critics response. Returns an
    EvaluationResult that tells the Coordinator what to do next.

    LEARNING NOTE:
      The two stopping conditions are independent:
        1. Quality gate  — the latest draft is good enough (few issues)
        2. Iteration cap — we've used all our rounds regardless of quality
      Condition 2 is a safety net. Without it, a demanding Critics could
      block the system indefinitely on a query that's inherently ambiguous.

    Args:
        candidates: All Candidate objects produced so far, in round order.

    Returns:
        EvaluationResult with should_iterate=False when ready to finalize.

    Example:
        >>> result = evaluate([round1, round2])
        >>> result.should_iterate
        False
        >>> result.reason
        'Draft is good enough (1 issue <= threshold 1)'
    """
    if not candidates:
        raise ValueError("evaluate() called with no candidates")

    latest = candidates[-1]
    best = best_candidate(candidates)

    # Stopping condition 1: iteration cap reached
    if latest.round >= MAX_ROUNDS:
        return EvaluationResult(
            should_iterate=False,
            reason=f"Reached max rounds ({MAX_ROUNDS}), taking best candidate",
            best=best,
        )

    # Stopping condition 2: quality gate passed
    if latest.critique.is_clean(max_issues=MAX_ISSUES_TO_PASS):
        return EvaluationResult(
            should_iterate=False,
            reason=(
                f"Draft is good enough "
                f"({latest.issue_count()} issue(s) <= threshold {MAX_ISSUES_TO_PASS})"
            ),
            best=latest,
        )

    # Otherwise: keep iterating
    return EvaluationResult(
        should_iterate=True,
        reason=(
            f"Round {latest.round}: {latest.issue_count()} issue(s) found, "
            f"asking Planner to revise"
        ),
        best=latest,
    )


# ---------------------------------------------------------------------------
# HANDS-ON: Run this file directly and trace the iteration loop
# ---------------------------------------------------------------------------
# Command: uv run python docs/evaluation.py
#
# The loop below simulates what the Coordinator does in Phase 4.
# Change MAX_ROUNDS or MAX_ISSUES_TO_PASS above and re-run to see the effect.

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
