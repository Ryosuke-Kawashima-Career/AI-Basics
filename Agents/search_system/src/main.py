from __future__ import annotations
 
import asyncio  # NOT `_asyncio` — that's the private C accelerator module
                 # and does not reliably expose `.run()` as public API.
import sys
import time

import httpx
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.coordinator.agent import coordinator_agent

SPECIALIST_URLS = {
    "websearcher": "http://localhost:8101/.well-known/agent-card.json",
    "planner": "http://localhost:8102/.well-known/agent-card.json",
    "critics": "http://localhost:8103/.well-known/agent-card.json",
}

def wait_for_agent(name: str, url: str, timeout: float = 10.0) -> None:
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
    raise RuntimeError(f"Agent at {url} did not start in time.")

def wait_for_all_specialitsts() -> None:
    """Block until every specialit in the list responds"""
    print("Waiting for specialist services.")
    for name, url in SPECIALIST_URLS.items():
        wait_for_agent(name, url)
    print("All specialists are reachable.\n")

async def ask(query: str) -> str:
    runner = InMemoryRunner(agent=coordinator_agent)
    final_text = ""
    message = types.UserContent(query)
    async for event in runner.run_async(user_id="cli", session_id="cli", new_message=message):
        if event.is_final_response():
            final_text = event.content.parts[0].text
    return final_text

def main():
    query = " ".join(sys.argv[1:])
    wait_for_all_specialitsts()
    print(f"Query: {query}\n")
    answer = asyncio.run(ask(query))
    print("===Final Answer===")
    print(answer)

if __name__ == "__main__":
    main()
          