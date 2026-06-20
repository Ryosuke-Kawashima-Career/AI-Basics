# A2A & Multi-Agent Architecture — Practice Questions

This set covers how A2A (Agent2Agent) connects independent agent processes, and how that fits into Phase 5 of your deep search system: wiring `src/main.py` to launch Websearcher (8101), Planner (8102), Critics (8103), and the Coordinator (8100) together, then sending a query through to a final answer.

Try answering each question before reading the explanation. Several questions reference real bugs you've already hit in this project (mismatched ports, stale agent definitions) — that's intentional, since debugging is where this architecture actually clicks.

## Key terms at a glance

- **A2A (Agent2Agent) protocol** — an open protocol that lets independent agent processes discover and call each other over HTTP, regardless of what framework built them. The "remote" in the name is literal: even on `localhost`, it's a real network call.
- **Agent card** — a JSON document describing what an agent can do (`name`, `description`, `skills`, `url`). It's the entry point for discovery, served by convention at `/.well-known/agent-card.json`.
- **`to_a2a()`** — the ADK helper you used in Phase 3. It wraps a local `LlmAgent` into a Starlette app, exposes it over HTTP, and auto-generates its agent card from the agent's own `name`/`description`/`instruction`.
- **`RemoteA2aAgent`** — the ADK class that wraps a *remote* agent's URL so it can be used as a `sub_agent`, just like a local one. It runs no model itself — it forwards requests over HTTP and translates the response back into ADK's native `Event` objects.
- **`sub_agents`** — the list of agents (local or remote) a parent `LlmAgent` can delegate to. The parent's `instruction` tells its LLM when to delegate to which sub-agent.
- **Task store** — an in-memory tracker that holds the state of in-flight A2A requests on the server side. `to_a2a()` creates one automatically; you don't manage it directly.
- **Process boundary** — the line between independent OS processes. A2A exists specifically because the Coordinator and each specialist run as *separate* processes on separate ports, not as Python objects inside one process.

---

## Part 1: The Rules

**Q1.** In plain terms, what problem does A2A solve that simply writing `from agents.websearcher.websearcher import websearcher_agent` inside `coordinator.py` wouldn't solve? Use your project's three specialists as the example.

*Explanation:* If the Coordinator directly imported `websearcher_agent`, it would be calling a Python object living in the *same process*. That works fine for a single-machine toy project, but it collapses the moment you want to: scale Websearcher independently (it does I/O-bound search calls, so it benefits from more replicas than Planner), deploy specialists on different machines, or let a completely different team's agent (built in a different framework) act as your Websearcher. A2A solves this by making each specialist a network service with a stable contract (the agent card) — the Coordinator calls `http://localhost:8101` and doesn't care whether that's a Python ADK agent, a Java agent, or something hosted on another continent. In your project specifically, this is *why* Phase 3 had you run `uv run uvicorn agents.websearcher.server:a2a_app --port 8101` as a standalone process instead of just importing it.

---

**Q2.** When you ran `to_a2a(websearcher_agent, port=8101)` in Phase 3, what did that single call actually set up for you, and where can you observe the result of each piece at runtime?

*Explanation:* `to_a2a()` does three things behind the scenes: (1) it wraps `websearcher_agent` in an `A2aAgentExecutor`, the bridge that translates incoming A2A HTTP requests into ADK's internal `Event` system — you can observe this indirectly by sending a request and getting a sensible response back; (2) it creates an `InMemoryTaskStore` to track in-flight requests — invisible unless you inspect server internals, but it's why concurrent requests to the same specialist don't clobber each other; (3) it builds a Starlette app and auto-generates an agent card from `websearcher_agent`'s `name`, `description`, and `instruction` — you can observe this directly by curling `http://localhost:8101/.well-known/agent-card.json` and seeing your agent's description reflected back as JSON.

---

**Q3.** An agent card lists a `description` and a list of `skills`, but never the full `instruction` string you wrote on the `LlmAgent`. Why the gap?

*Explanation:* The agent card is meant to be a lightweight, public *contract* — just enough for another agent (or a human) to decide "should I call this one?" The full `instruction` string often contains internal reasoning steps, formatting rules, or edge-case handling that's irrelevant (or even confusing) to an outside caller, and may be long enough to bloat every discovery request if included. This is the same reason a REST API publishes an OpenAPI summary rather than its server-side source code: the consumer needs the *interface*, not the implementation. Practically, this means the `description=` you set on each `LlmAgent` in Phase 2 is doing double duty — it's both documentation for you and the actual text the Coordinator's LLM will read when deciding whether to delegate to that specialist. A vague description (e.g., `"A helpful agent"`) makes that decision harder for the Coordinator, even though the underlying `instruction` might be excellent.

---

## Part 2: Applying It

**Q4.** Write the wiring code that lets the Coordinator treat the already-running Websearcher service (port 8101) as one of its `sub_agents`, using `RemoteA2aAgent`.

*Explanation:*
```python
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent

websearcher_remote = RemoteA2aAgent(
    name="websearcher",
    description="Gathers background information and sources for a research question.",
    agent_card=f"http://localhost:8101{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)
```
Three things matter here: `name` is the local handle the Coordinator's instruction logic will refer to; `description` should closely mirror what `websearcher_agent` itself declares (it's read by the Coordinator's LLM, so a mismatch here actively misleads it); and `agent_card` is the *full URL to the card*, not just the base URL — `AGENT_CARD_WELL_KNOWN_PATH` is ADK's constant for `/.well-known/agent-card.json`, so you don't have to hardcode that path string everywhere. `use_legacy=False` opts into the newer A2A executor (the one that supports request interceptors and custom converters, if you need them later).

---

**Q5.** The Coordinator declares `sub_agents=[websearcher_remote, planner_remote, critics_remote]`. Trace what actually happens, step by step, when a user asks "Explain RAG vs fine-tuning."

*Explanation:* (1) The Coordinator's own `LlmAgent` receives the query and reads its own `instruction` plus the `description` of each sub-agent. (2) Its underlying LLM decides — purely from those descriptions, the same way it'd choose between any tools — that it needs background research first, so it emits a delegation targeting `websearcher_remote`. (3) `RemoteA2aAgent` converts that delegation into an actual HTTP POST to `http://localhost:8101`. (4) The Websearcher's `A2aAgentExecutor` receives it, runs `websearcher_agent`, and returns a response. (5) `RemoteA2aAgent` converts that HTTP response back into an ADK `Event`, which gets added to the Coordinator's context. (6) The Coordinator's LLM now has the research in context and decides to delegate to `planner_remote` next, repeating the cycle. **Important nuance:** this LLM-driven delegation via `sub_agents` is a *different mechanism* than the deterministic `intake.py → candidates.py → evaluation.py → recommendation.py` pipeline you scaffolded in Phase 4. If you want the guaranteed, code-controlled iteration loop (cap at `MAX_ROUNDS`, structured `Critique` objects), your Coordinator's Python code should call each `RemoteA2aAgent`'s `run()` method explicitly inside that pipeline, rather than relying on the LLM to decide the order via free-form delegation. Both are valid A2A usage — but they give you very different levels of control, and it's worth deciding now which one your Coordinator actually uses.

---

**Q6.** Why can't `main.py` just call `websearcher_agent.run(...)` directly in-process, now that you've committed to A2A?

*Explanation:* Once `RemoteA2aAgent` is configured with `agent_card="http://localhost:8101/..."`, that URL is the *only* thing it knows how to reach — it has no fallback to an in-process object, because it was never given one. If the specialist server isn't actually listening on 8101 when the Coordinator tries to delegate, `RemoteA2aAgent` makes a real HTTP call that gets `ConnectionRefusedError`. This is the same failure mode as any microservice architecture: the abstraction (treat a sub-agent like a function call) only holds if the underlying service is actually running. `main.py`'s job in Phase 5 is precisely to guarantee that ordering — start every specialist server *before* the Coordinator ever receives a query.

---

## Part 3: Where It Shows Up in Practice (Phase 5)

**Q7.** Sketch, in order, what `src/main.py` needs to do for `make run` to work end to end. Where does it need to *wait* before moving to the next step?

*Explanation:* The sequence is: (1) launch the three specialist servers as subprocesses (`uvicorn agents.websearcher.server:a2a_app --port 8101`, and similarly for 8102/8103); (2) **wait** — poll each specialist's `/.well-known/agent-card.json` endpoint with retries until it responds, because starting the Coordinator before its `RemoteA2aAgent` targets are reachable just defers Q6's `ConnectionRefusedError` to query time instead of catching it at startup; (3) launch the Coordinator (its own `to_a2a()`-wrapped server, or `adk web`, depending on how you want to expose it); (4) optionally launch `adk web` pointed at the `agents/` directory for interactive testing. A minimal health-check loop looks like:
```python
import time
import httpx

def wait_for_agent(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Agent at {url} did not start in time")
```
The "wait" step is the part people skip and then get confused by intermittent startup failures — it's the direct fix for Q6's failure mode.

---

**Q8.** Where does ADK Web (port 8080) fit into this picture — is it another A2A participant, or something different?

*Explanation:* ADK Web is a developer UI, not an A2A service. It loads agents from your `agents/` directory and lets you chat with whichever one you select — in your case, the Coordinator — the same way a human user would through any chat interface. It doesn't get an agent card and the Coordinator doesn't delegate to it. It's orthogonal to the Websearcher/Planner/Critics A2A wiring: useful for interactive testing in Phase 6, but the Coordinator works identically whether or not ADK Web is running. Think of it as a browser pointed at your root agent, not a node in the agent graph.

---

**Q9.** You want a non-interactive way to test the system — no browser, just a script that sends one query and prints the final answer. What does that CLI entry in `main.py` need to do?

*Explanation:* This uses the same underlying mechanism ADK Web uses internally — an ADK `Runner` wrapping your root agent (the local Coordinator `LlmAgent` object, *not* a `RemoteA2aAgent`, since you're running in-process here) — just invoked programmatically instead of through a chat UI:
```python
import asyncio
from google.adk.runners import InMemoryRunner

from agents.coordinator.agent import coordinator_agent

async def ask(query: str) -> str:
    runner = InMemoryRunner(agent=coordinator_agent)
    final_text = ""
    async for event in runner.run_async(user_id="cli", session_id="cli", new_message=query):
        if event.is_final_response():
            final_text = event.content.parts[0].text
    return final_text

if __name__ == "__main__":
    print(asyncio.run(ask("Explain RAG vs fine-tuning")))
```
This is the Phase 5 deliverable from the implementation plan: "a simple CLI... so a query can be sent to the Coordinator and the final answer printed." Note that the Coordinator itself still reaches Websearcher/Planner/Critics over A2A internally — only the *outermost* call (your script to the Coordinator) is in-process here.

---

## Part 4: Debugging

**Q10.** `main.py` starts, all three specialists come up cleanly, but the Coordinator throws `httpx.ConnectError` the first time it tries to delegate to `planner_remote`. Planner's `server.py` looks correct on inspection. What two things do you check first, in order?

*Explanation:* First, confirm the Planner server process is *actually still alive* and bound to 8102 — `curl http://localhost:8102/.well-known/agent-card.json` directly, bypassing the Coordinator entirely. A server can crash silently after startup (you saw exactly this pattern earlier in this project, when `websearcher/server.py` had a leftover `from a2a.server import A2AServer` import alongside the correct `to_a2a()` line — the bad import broke the module on load, so nothing was ever listening on 8101 even though the file "looked" mostly right). Second, *if* curl succeeds independently, check that the `agent_card=` URL passed into `planner_remote`'s `RemoteA2aAgent` definition actually says `8102` and not `8101` — copy-pasting the Websearcher wiring three times and forgetting to update the port number is a very common bug, and it produces exactly this symptom: the Coordinator is connecting successfully, just to the wrong specialist's port (or to a service that genuinely isn't there).

---

**Q11.** Websearcher's agent card loads fine at its well-known URL, but its `description` field shows up as a generic `"An AI agent"` instead of the description you wrote. What's the most likely cause, and which file do you check first?

*Explanation:* Check `agents/websearcher/websearcher.py` first, not `server.py`. `to_a2a()` builds the agent card *from the agent object itself* — it reads `websearcher_agent.description` (and `name`, `instruction`) at the moment the card is generated. A generic fallback description almost always means the `LlmAgent(...)` constructor never actually had `description=` set (or it was set on a different, stale agent object that `server.py` isn't importing). This is a good example of where the bug *looks* like it's in the A2A plumbing (the card!) but is actually upstream in the agent definition — the card is just an honest mirror of whatever the agent object declares.

---

## Part 5: Stretch

**Q12.** Right now, Coordinator calls Websearcher, then Planner, then Critics — three sequential HTTP round trips. If you had five specialists instead of three, and some of them had no dependency on each other's output, how could you reduce total latency? What ADK construct hints at this?

*Explanation:* Sequential calls are only required when each step genuinely depends on the previous one's output — Critics needs Planner's draft, so that dependency is real. But if, say, two independent specialists both only need the original user query (not each other's output), you could fire both `RemoteA2aAgent` calls concurrently with `asyncio.gather` and await both before proceeding, cutting that portion of latency roughly in half. ADK's docs hint at a more structured version of this with `ParallelAgent`, a workflow-agent construct designed for exactly this case — running multiple sub-agents concurrently and merging their outputs, rather than relying on the LLM to decide one-at-a-time. The trade-off: parallelism only helps where the dependency graph genuinely branches. Force-parallelizing Critics before Planner has even produced a draft wouldn't save time — it would just mean Critics evaluates nothing.

---

### Tip for self-study

Before wiring any two agents together with `RemoteA2aAgent`, `curl` the remote agent's `/.well-known/agent-card.json` by hand first. If `curl` can't see it, neither can `RemoteA2aAgent` — and that one check instantly tells you whether you're looking at a wiring bug (wrong port, wrong description) or an environment bug (server not running, crashed on import). You've already used this technique once in this project; it's worth making it a reflex before debugging anything further upstream.
