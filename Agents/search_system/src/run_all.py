"""
EXAMPLE: src/run_all.py — one command to start everything
=============================================================
PURPOSE: Launch the three specialist A2A services (Websearcher, Planner,
         Critics) as background subprocesses, wait until all three are
         reachable, then run the same query flow as `src/main.py` — all
         from a single command instead of four separate terminals.

WHY THIS FILE EXISTS:
  Manually opening 4 terminals (3 specialists + main.py) works, but it's
  tedious and easy to get wrong — forget to start one, or start them in
  the wrong order. This script automates exactly what `make run` would do
  once you have a Makefile: start every specialist process, confirm they're
  up, then drive the Coordinator. On shutdown (Ctrl+C), it cleans up every
  subprocess it started so you don't end up with orphaned servers still
  holding ports 8101-8103.

HOW TO RUN — must be invoked as a module, from the project root:
    uv run python -m src.run_all "Explain RAG vs fine-tuning"

  Do NOT run it as `uv run python src/run_all.py` — see the
  ModuleNotFoundError note below for why that fails.

  With no argument, it starts an interactive loop where you can type
  multiple queries; type `exit` or press Ctrl+C to shut everything down.

PREREQUISITE:
  This still assumes `agents/coordinator/agent.py` exists and exports a
  `coordinator_agent` LlmAgent with its sub_agents wired via RemoteA2aAgent
  (see the architecture note in docs/main.py if that file isn't built yet).

WHY `ModuleNotFoundError: No module named 'agents'` HAPPENS HERE:
  There's no `agents/__init__.py` or packaging config in this project —
  `agents` and `src` are just plain directories. They only become
  importable when the project ROOT is on `sys.path`. Two ways that happens:
    - `python -m src.run_all`  -> Python adds the current *working*
      directory (the root, if you run it from there) to sys.path.
    - `python src/run_all.py` -> Python adds the *script's own folder*
      (`src/`) to sys.path instead — the root is never added, so `agents`
      can't be found, even though it sits right next to `src/`.
  The sys.path.insert below is a defensive fallback so this still works
  even if someone runs it the second way (e.g. an IDE "Run" button).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Defensive fallback: guarantee the project root is importable regardless
# of how this file was launched (see WHY note above).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import subprocess
import time

import httpx
from google.adk.runners import InMemoryRunner
from google.genai import types

# ---------------------------------------------------------------------------
# Step 1: Define what to launch
# ---------------------------------------------------------------------------
# Each entry is (display_name, module:app target for uvicorn, port).
# Add a 4th tuple here if you add a 4th specialist — nothing else in this
# file needs to change.

SPECIALISTS = [
    ("websearcher", "agents.websearcher.server:a2a_app", 8101),
    ("planner", "agents.planner.server:a2a_app", 8102),
    ("critics", "agents.critics.server:a2a_app", 8103),
]


def agent_card_url(port: int) -> str:
    return f"http://localhost:{port}/.well-known/agent-card.json"


# ---------------------------------------------------------------------------
# Step 2: Start each specialist as a background subprocess
# ---------------------------------------------------------------------------
# We use `sys.executable -m uvicorn ...` (rather than shelling out to the
# `uv run uvicorn ...` command) so the child process reuses the exact same
# Python interpreter and virtual environment this script is already running
# under — no dependency on `uv` being resolvable inside the subprocess call.
# Child processes inherit this console's stdout/stderr, so each specialist's
# own logs (and any startup errors) print directly to your terminal,
# interleaved — useful for noticing if one of them crashes on import, the
# exact failure mode you hit earlier with a stale `A2AServer` import.

def start_specialists() -> list[subprocess.Popen]:
    processes = []
    for name, target, port in SPECIALISTS:
        print(f"Starting {name} on port {port}...")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", target,
                "--host", "localhost", "--port", str(port),
            ]
        )
        processes.append(proc)
    return processes


def stop_specialists(processes: list[subprocess.Popen]) -> None:
    print("\nShutting down specialists...")
    for proc in processes:
        proc.terminate()
    for proc in processes:
        proc.wait(timeout=5)
    print("All specialists stopped.")


# ---------------------------------------------------------------------------
# Step 3: Wait until every specialist responds before touching the Coordinator
# ---------------------------------------------------------------------------

def wait_for_agent(name: str, url: str, timeout: float = 15.0) -> None:
    """Poll until `url` returns 200, or raise after `timeout` seconds.

    LEARNING NOTE: the `raise` sits AFTER the while loop, not inside it —
    so it only fires once the deadline has actually passed, not on the
    first failed attempt. See docs/main.py for the version of this bug
    that DOES raise too early, and why that's wrong.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=10.0).status_code == 200:
                print(f"  [ok] {name} is reachable")
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{name} did not become reachable at {url} within {timeout}s")


def wait_for_all_specialists() -> None:
    print("Waiting for specialists to come online...")
    for name, _target, port in SPECIALISTS:
        wait_for_agent(name, agent_card_url(port))
    print("All specialists are up.\n")


# ---------------------------------------------------------------------------
# Step 4: Send a query to the Coordinator (in-process)
# ---------------------------------------------------------------------------
# Imported lazily, inside main(), rather than at module load time — if
# `agents/coordinator/agent.py` has a bug, you want the specialists to have
# already started and been confirmed reachable before that import error
# surfaces, so you can tell the two failure modes apart.

async def ask(coordinator_agent, query: str) -> str:
    runner = InMemoryRunner(agent=coordinator_agent)
    final_text = ""
    message = types.UserContent(query)
    async for event in runner.run_async(user_id="cli", session_id="cli", new_message=message):
        if event.is_final_response():
            final_text = event.content.parts[0].text
    return final_text


# ---------------------------------------------------------------------------
# Step 5: Tie it all together
# ---------------------------------------------------------------------------

def main() -> None:
    processes = start_specialists()
    try:
        wait_for_all_specialists()

        from agents.coordinator.agent import coordinator_agent  # see prerequisite note above

        initial_query = " ".join(sys.argv[1:])
        if initial_query:
            # Single-shot mode: one query from the command line, then exit.
            print(f"Query: {initial_query}\n")
            answer = asyncio.run(ask(coordinator_agent, initial_query))
            print("=== Final Answer ===")
            print(answer)
        else:
            # Interactive mode: keep the specialists up across multiple queries.
            print("Interactive mode. Type a query, or 'exit' to quit.\n")
            while True:
                query = input("> ").strip()
                if query.lower() in {"exit", "quit"}:
                    break
                if not query:
                    continue
                answer = asyncio.run(ask(coordinator_agent, query))
                print(f"\n{answer}\n")
    except KeyboardInterrupt:
        pass
    finally:
        stop_specialists(processes)


if __name__ == "__main__":
    main()
