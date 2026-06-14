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


# ---------------------------------------------------------------------------
# FinalAnswer
# ---------------------------------------------------------------------------
# The typed output of this module — what the Coordinator returns to the user.

class FinalAnswer:
    def __init__(self, text: str, sources: list[dict], rounds_taken: int):
        self.text = text                  # The formatted answer text
        self.sources = sources            # List of {title, url} dicts
        self.rounds_taken = rounds_taken  # How many Planner→Critics cycles ran

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
# These are pure functions — no LLM calls, no side effects. Easy to unit test.

def enforce_word_limit(text: str, max_words: int) -> str:
    """Trim text to at most `max_words` words, cutting at a sentence boundary.

    LEARNING NOTE:
      Cutting at a sentence boundary ('. ') is friendlier than hard-cutting
      at a word index. If no sentence boundary is found before the limit,
      it falls back to word-level trimming. A more sophisticated version
      could call an LLM to summarize rather than truncate.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Try to end at a sentence boundary for cleaner output
    last_sentence_end = truncated.rfind(". ")
    if last_sentence_end > len(truncated) // 2:
        return truncated[: last_sentence_end + 1]
    return truncated + "…"


def format_sources(sources: list[dict]) -> str:
    """Render a sources list as a numbered citation block.

    Expects each source as {"title": str, "url": str}.

    Example:
        >>> format_sources([{"title": "RAG paper", "url": "https://arxiv.org/..."}])
        'Sources:\\n[1] RAG paper — https://arxiv.org/...'
    """
    if not sources:
        return ""
    lines = [
        f"[{i}] {s.get('title', 'Untitled')} — {s.get('url', '')}"
        for i, s in enumerate(sources, start=1)
    ]
    return "Sources:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Core recommendation function
# ---------------------------------------------------------------------------

def build_recommendation(
    candidate: Candidate,
    query: ParsedQuery,
    sources: list[dict],
) -> FinalAnswer:
    """Produce the final answer from the winning Candidate.

    Applies constraints from `query` (word limit, formatting) and appends
    a citation block from `sources` (gathered by the Websearcher).

    LEARNING NOTE:
      This function is the last step before the answer leaves the Coordinator.
      The order of operations matters:
        1. Apply word limit BEFORE appending citations — so citations don't
           get cut off.
        2. Append citations AFTER — so the word limit only applies to the
           answer body, not the source list.

    Args:
        candidate: The winning Candidate from evaluation.py.
        query:     The ParsedQuery from intake.py (carries constraints).
        sources:   Raw search results from the Websearcher.

    Returns:
        FinalAnswer ready to return to the user.
    """
    answer_text = candidate.draft

    # Apply word limit if the user asked for one
    if query.max_words is not None:
        answer_text = enforce_word_limit(answer_text, query.max_words)

    # Append citation block
    citation_block = format_sources(sources)
    if citation_block:
        answer_text = f"{answer_text}\n\n{citation_block}"

    return FinalAnswer(
        text=answer_text,
        sources=sources,
        rounds_taken=candidate.round,
    )


# ---------------------------------------------------------------------------
# HANDS-ON: Run this file directly
# ---------------------------------------------------------------------------
# Command: uv run python docs/recommendation.py
#
# Try changing max_words in the ParsedQuery below to 20 and see what gets cut.
# Then try setting it to None to see the full answer with citations.

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
