# Implementation Plan: Deep Search Agent System

Based on `specs/search_design.md`. Target stack: Python 3.11, Google ADK + A2A, `uv`/Makefile per repo conventions.

## Phase 0 — Project scaffolding

1. Run `make setup` to install dependencies via `uv` (confirms `pyproject.toml`/`uv.lock` are in place).
2. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` (or Agent Platform vars from README).
3. Create the directory skeleton from the design doc:
   - `src/main.py`
   - `agents/coordinator/coordinator.py`
   - `agents/websearcher/websearcher.py`
   - `agents/planner/planner.py`
   - `agents/critics/critics.py`
   - `agents/tools.py`

*Why first:* everything else imports from `agents/tools.py` and runs through `src/main.py`, so the skeleton needs to exist before any agent logic.

## Phase 1 — Shared tools (`agents/tools.py`)

1. Define the web search tool the Websearcher will call (e.g., wrap a search API or existing ADK `google_search` tool).
2. Define any shared helper functions (formatting search results, truncating context, citation tagging).
3. Keep these as small, pure functions — per `CLAUDE.md`, business logic belongs in `src/hirenest_support/`-style modules, not buried in agent wiring.

*Concrete example:* a `web_search(query: str) -> list[dict]` function that returns `[{"title", "url", "snippet"}, ...]`, independently testable without spinning up an agent.

## Phase 2 — Leaf specialist agents

Build these three first since they have no dependencies on each other — only the Coordinator depends on them.

1. **Websearcher** (`agents/websearcher/websearcher.py`)
   - ADK `LlmAgent` wired to the `web_search` tool from `tools.py`.
   - Input: a research question from the Coordinator. Output: condensed background findings with sources.
2. **Planner** (`agents/planner/planner.py`)
   - ADK `LlmAgent` that drafts an initial answer to the user's query, optionally consuming Websearcher output.
3. **Critics** (`agents/critics/critics.py`)
   - ADK `LlmAgent` that takes the Planner's draft and returns a structured critique (issues found, missing evidence, suggested revisions).

*Concrete example:* give the Planner a query like "What are the trade-offs of vector vs. keyword search?" and verify it produces a draft; then feed that draft to Critics and check it returns concrete, actionable critique points rather than vague praise.

## Phase 3 — Expose specialists over A2A

1. Wrap each of Websearcher, Planner, and Critics as A2A services (per `make run-specialists`, ports `8101`–`810X`).
2. Verify each one responds independently — e.g., `curl` or a small test script hitting each port with a sample request.

*Why this order:* the Coordinator talks to these as remote A2A agents, not as in-process objects, so they must be independently runnable before wiring the Coordinator.

## Phase 4 — Coordinator (`agents/coordinator/coordinator.py`)

This is the most complex piece — split it into the sub-files the design implies (`agent.py`, `intake.py`, `candidates.py`, `evaluation.py`, `recommendation.py`, prompts/models):

1. `intake.py` — parse and normalize the user's query.
2. Orchestration loop — call Websearcher for background, then Planner for an initial answer, then Critics to evaluate it.
3. `evaluation.py` / `candidates.py` — structure for comparing the Planner's draft against the Critics' feedback (possibly iterating: replan → re-critique).
4. `recommendation.py` — produce the final decision/answer returned to the user, synthesizing Planner output + Critic feedback.
5. Wire it to run on port `8100` (`make run-coordinator`).

*Concrete example:* trace one query end-to-end on paper first — "Explain RAG vs. fine-tuning" → Coordinator asks Websearcher for background → Planner drafts an answer → Critics flags that it omits cost trade-offs → Coordinator either asks Planner to revise or synthesizes the final answer itself, citing Websearcher's sources.

## Phase 5 — Entry point (`src/main.py`)

1. Wire up startup: launch specialists + Coordinator + ADK Web (mirrors `make run`).
2. Add a simple CLI or programmatic entry so a query can be sent to the Coordinator and the final answer printed.

## Phase 6 — Lint, test, and run end-to-end

1. `make lint` — Ruff checks (rules `E`, `F`, `I`, `UP`, `B`; 100-char lines).
2. `make web` — exercise the system interactively through ADK Web on port `8080`.
3. `make run` — full end-to-end smoke test: specialists + coordinator + ADK Web together.
4. Since there's no test suite in this stripped-down example, write a couple of manual smoke-test queries (varying complexity) and confirm the Coordinator returns a coherent, sourced final answer.

## Suggested build order recap

`tools.py` → Websearcher → Planner → Critics → A2A wrapping → Coordinator → `main.py` → lint/run

This order means each agent can be smoke-tested in isolation before being wired into the larger flow, which keeps debugging localized.

---

## Follow-up questions to deepen your understanding

1. The design diagram shows the Coordinator calling Planner, Websearcher, and Critics directly — but does the Planner ever need Websearcher's output *before* drafting, or only the Critics afterward? How would you change the flow if the Planner should research first?
2. What should the Coordinator do if the Critics keep finding problems after several revision rounds — loop forever, cap the iterations, or escalate to the user? How would you encode that decision in `evaluation.py`?
3. The Websearcher is the only agent with an external tool (`web_search`). What happens if that tool fails or returns nothing — should Planner and Critics proceed without background info, or should the Coordinator short-circuit?
4. A2A means each specialist runs as its own service on its own port. What's the trade-off versus just calling them as in-process Python functions — and why might the design doc's authors have chosen A2A here?
