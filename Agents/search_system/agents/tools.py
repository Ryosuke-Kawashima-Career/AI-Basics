"""Shared tools and helpers used by the specialist agents.

Keep functions here small and pure (per AGENTS.md) so they can be unit-tested
independently of any agent or LLM call. Agents in `agents/<name>/<name>.py`
should import from this module rather than redefining tool logic locally.
"""

from __future__ import annotations

from google.adk.tools import google_search

# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------
# `google_search` is ADK's built-in search tool — pass it directly into an
# LlmAgent's `tools=[...]` list (typically the Websearcher). Re-exporting it
# here keeps a single shared import path and makes it easy to swap in a
# different provider later without touching agent files.
web_search_tool = google_search


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------
# These are plain functions with no ADK/LLM dependency — easy to test with
# `pytest` or simple asserts, and reusable across Websearcher, Planner, and
# Critics wherever they need to shape search results or context.


def format_search_results(results: list[dict]) -> str:
    """Render raw search results into a compact, citation-friendly block.

    Expects each result to look like:
        {"title": str, "url": str, "snippet": str}

    Example:
        >>> format_search_results([
        ...     {"title": "RAG vs fine-tuning", "url": "https://example.com/a",
        ...      "snippet": "RAG retrieves context at query time..."},
        ... ])
        '[1] RAG vs fine-tuning (https://example.com/a)\\n    RAG retrieves context...'
    """
    lines = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        lines.append(f"[{i}] {title} ({url})\n    {snippet}")
    return "\n".join(lines)


def truncate_context(text: str, max_chars: int = 4000) -> str:
    """Trim long context blocks so prompts stay within budget.

    Cuts on a word boundary where possible and appends a marker so the
    model (and the developer reading logs) knows truncation happened.
    """
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].rstrip() + " [...truncated]"


def tag_citations(answer: str, sources: list[dict]) -> str:
    """Append a numbered source list to an answer for traceability.

    Pairs with `format_search_results`: Websearcher produces numbered
    results, and the Coordinator/Planner can call this to attach the same
    numbering as a "Sources" section on the final answer.
    """
    if not sources:
        return answer
    citation_lines = [
        f"[{i}] {s.get('title', 'Untitled')} — {s.get('url', '')}"
        for i, s in enumerate(sources, start=1)
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(citation_lines)
