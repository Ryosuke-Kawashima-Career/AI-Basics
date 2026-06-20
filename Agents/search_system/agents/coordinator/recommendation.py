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
from typing import List, Dict

from candidates import Candidate
from intake import ParsedQuery

class FinalAnswer:
    def __init__(self, text: str, sources: List[str], rounds_taken: int) -> None:
        self.text = text
        self.sources = sources
        self.rounds_taken = rounds_taken
    
    def __str__(self) -> str:
        return self.text

def enforce_word_limit(text: str, max_words: int) -> str:
    """Trim texts to at most "max_words", cutting at a sentence boundary"""
    words = text.split()
    if len(words) == max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_sentence_end = truncated.rfind(". ")
    if last_sentence_end > len(truncated) // 2:
        return truncated[: last_sentence_end + 1]
    return truncated + "..."

def format_sources(sources: List[Dict]) -> str:
    """Renders a source list as a numbered citation block"""
    if not sources:
        return ""
    lines = [
        f"[{i}] {s.get('title', 'Untitled')}-{s.get('url', '')}"
        for i, s in enumerate(sources, start=1)
    ]
    return "Sources:\n" + ".\n".join(lines)

def build_recommendation(
      candidate: Candidate,
      query: ParsedQuery,
      sources: List[Dict],
) -> FinalAnswer:
    """Produces the final answer from the winning candidate
    Applies constraints from query (word limit, formatting) and appends a citation block from the sources 
    """
    answer_text = candidate.draft
    if query.max_words is not None:
        answer_text = enforce_word_limit(answer_text, query.max_words)
    citation_block = format_sources(sources)
    if len(citation_block) > 0:
        answer_text = f"{answer_text}\n\n{citation_block}"
    return FinalAnswer(
        text=answer_text,
        sources=sources,
        rounds_taken=candidate.round,
    )

if __name__ == "__main__":
    from candidates import Critique  # noqa: PLC0415
 
    winning_candidate = Candidate(
        round=2,
        draft=(
            "RAG retrieves context at query time, making it cheaper to update "
            "when your data changes frequently. Fine-tuning bakes knowledge into "
            "model weights, giving faster inference but requiring full retraining "
            "to update. Hybrid approaches combine both for high accuracy and "
            "freshness. Choose RAG for frequently-changing data; fine-tuning for "
            "stable domains where latency matters."
        ),
        critique=Critique(
            issues=["Could mention hybrid approaches"],
            missing_evidence=[],
            suggested_revisions=[],
        ),
    )
 
    parsed_query = ParsedQuery(
        question="What are the trade-offs between RAG and fine-tuning?",
        constraints=["concise"],
        specialists_needed=["websearcher", "planner", "critics"],
        max_words=50,  # ← change this to None or 200 to see the difference
    )
 
    websearcher_sources = [
        {"title": "RAG vs Fine-tuning Survey", "url": "https://arxiv.org/abs/2312.10997"},
        {"title": "Fine-tuning LLMs: Cost Analysis", "url": "https://openai.com/research/..."},
    ]
 
    result = build_recommendation(winning_candidate, parsed_query, websearcher_sources)
 
    print("=== FINAL ANSWER ===")
    print(result)
    print(f"\n(Produced in {result.rounds_taken} round(s))")
