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
