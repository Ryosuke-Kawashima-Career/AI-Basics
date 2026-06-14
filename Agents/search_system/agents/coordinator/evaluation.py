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
