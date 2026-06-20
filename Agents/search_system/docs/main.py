"""
EXAMPLE: src/main.py (Phase 5 — Entry point)
==============================================
PURPOSE: Wait for the three specialist A2A services to be reachable, then
         send one query into the Coordinator (in-process) and print the
         final answer.

WHY THIS FILE EXISTS:
  Phase 0-4 built the pieces in isolation: three specialists each runnable
  on their own port, and a Coordinator pipeline (intake/candidates/
  evaluation/recommendation) testable with hand-built data. main.py is
  where those pieces actually meet for the first time — it is the only
  file in the whole project that assumes ALL specialist processes are
  already running.

PREREQUISITES (start these in separate terminals BEFORE running this file):
    uv run uvicorn agents.websearcher.server:a2a_app --port 8101
    uv run uvicorn agents.planner.server:a2a_app     --port 8102
    uv run uvicorn agents.critics.server:a2a_app     --port 8103

ARCHITECTURE NOTE — where does RemoteA2aAgent wiring belong?
  It belongs in `agents/coordinator/agent.py`, NOT in main.py. main.py's
  only job is process orchestration (wait for services, send one query,
  print the result). The Coordinator's `agent.py` should look like:

      from google.adk.agents import LlmAgent
      from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent

      websearcher_remote = RemoteA2aAgent(
          name="websearcher",
          description="Gathers background information and sources for a research question.",
          agent_card=f"http://localhost:8101{AGENT_CARD_WELL_KNOWN_PATH}",
          use_legacy=False,
      )
      planner_remote = RemoteA2aAgent(
          name="planner",
          description="Drafts an initial answer to the user's query.",
          agent_card=f"http://localhost:8102{AGENT_CARD_WELL_KNOWN_PATH}",
          use_legacy=False,
      )
      critics_remote = RemoteA2aAgent(
          name="critics",
          description="Critically evaluates the Planner's draft answer.",
          agent_card=f"http://localhost:8103{AGENT_CARD_WELL_KNOWN_PATH}",
          use_legacy=False,
      )

      coordinator_agent = LlmAgent(
          name="coordinator",
          model="gemini-2.0-flash",
          description="Coordinates research, drafting, and critique to answer the user.",
          instruction="...",
          sub_agents=[websearcher_remote, planner_remote, critics_remote],
      )

  Keeping the RemoteA2aAgent definitions in agent.py (not main.py) means
  main.py never needs to change when you add a 4th specialist — it just
  imports whatever `coordinator_agent` object agent.py exposes.

  NOTE ON TWO ARCHITECTURES: the sketch above uses ADK's native LLM-driven
  `sub_agents` delegation. Your `evaluation.py` pipeline from Phase 4 is a
  SEPARATE, deterministic alternative (explicit calls + an iteration cap).
  Pick one as your real Coordinator — mixing both in the same agent is
  possible but confusing while you're still learning the architecture.
"""

from __future__ import annotations

import asyncio  # NOT `_asyncio` — that's the private C accelerator module
                 # and does not reliably expose `.run()` as public API.
import sys
import time

import httpx
from google.adk.runners import InMemoryRunner

from agents.coordinator.agent import coordinator_agent  # you still need to create this file

# ---------------------------------------------------------------------------
# Step 1: Know what you're waiting for
# ---------------------------------------------------------------------------
# Keep the specialist URLs in one place. If you add a 4th specialist, this
# is the only line in main.py that needs to change.

SPECIALIST_URLS = {
    "websearcher": "http://localhost:8101/.well-known/agent-card.json",
    "planner": "http://localhost:8102/.well-known/agent-card.json",
    "critics": "http://localhost:8103/.well-known/agent-card.json",
}


def wait_for_agent(name: str, url: str, timeout: float = 10.0) -> None:
    """Poll an agent card URL until it responds, or raise after `timeout` seconds.

    LEARNING NOTE — a common bug to watch for:
      The `raise` must sit OUTSIDE the while loop (after it exits), not
      inside it. If you accidentally indent `raise` to the same level as
      `time.sleep(...)`, the function raises on the very FIRST failed
      attempt instead of retrying until the deadline — which defeats the
      entire purpose of a retry loop. Compare this working version against
      your own implementation if something raises immediately.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                print(f"  [ok] {name} is up ({url})")
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"{name} did not become reachable at {url} within {timeout}s")


def wait_for_all_specialists() -> None:
    """Block until every specialist in SPECIALIST_URLS responds.

    Calling this before touching `coordinator_agent` turns a confusing
    mid-query `httpx.ConnectError` (Q6/Q10 in the A2A practice doc) into a
    clear, early failure that names exactly which specialist isn't running.
    """
    print("Waiting for specialist services...")
    for name, url in SPECIALIST_URLS.items():
        wait_for_agent(name, url)
    print("All specialists are reachable.\n")


# ---------------------------------------------------------------------------
# Step 2: Send one query to the Coordinator and collect the final answer
# ---------------------------------------------------------------------------
# This uses ADK's Runner — the same machinery `adk web` uses internally —
# just invoked programmatically instead of through a chat UI. The Coordinator
# itself is used IN-PROCESS here (we imported the LlmAgent object directly);
# it is the Coordinator's OWN sub_agents that go out over real A2A/HTTP to
# reach Websearcher, Planner, and Critics.

async def ask(query: str) -> str:
    """Send `query` to the Coordinator and return its final response text."""
    runner = InMemoryRunner(agent=coordinator_agent)
    final_text = ""
    async for event in runner.run_async(
        user_id="cli", session_id="cli", new_message=query
    ):
        if event.is_final_response():
            final_text = event.content.parts[0].text
    return final_text


# ---------------------------------------------------------------------------
# Step 3: CLI entry point
# ---------------------------------------------------------------------------
# Run with:  uv run python -m src.main "Explain RAG vs fine-tuning"
# Or with no argument, it falls back to a default sample query.

def main() -> None:
    query = " ".join(sys.argv[1:]) or "Explain RAG vs fine-tuning"

    wait_for_all_specialists()

    print(f"Query: {query}\n")
    answer = asyncio.run(ask(query))
    print("=== Final Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
