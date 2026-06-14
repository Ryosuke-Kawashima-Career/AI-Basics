"""
EXAMPLE: agents/coordinator/recommendation.py
==============================================
PURPOSE: Format the winning Candidate into the final answer the user sees —
         applying constraints from intake.py and attaching citations.
 
WHY THIS FILE EXISTS:
  The best Candidate from evaluation.py is still raw: its draft may be too
  long, has no citations attached, and may not respect constraints like
  "bullet points" or "under 150 words" from intake.py. recommendation.py
  is the single place where all those concerns come together before the
  answer leaves the Coordinator.
 
  Keeping this separate from evaluation.py means the decision of "which
  draft is best" is never mixed with the decision of "how to present it" —
  two different concerns that change for different reasons.
 
HOW TO USE THIS EXAMPLE:
  Read through, then try:
    1. Add a "bullet points" formatter that converts sentences to a list
    2. Make enforce_word_limit() trim from the end rather than the start
    3. Run the __main__ block and check the output matches your expectation
"""
 
from __future__ import annotations
 
from candidates import Candidate
from intake import ParsedQuery
from typing import List
from pydantic import BaseModel, Field
