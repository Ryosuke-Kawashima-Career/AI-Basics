"""
03_multiagent_supervisor_vis.py
================================
Architecture Pattern: Multi-Agent Supervisor / Router Pattern (Antigravity SDK & ADK Alignment)

Key Concepts:
1. Supervisor / Router Node: Analyzes tasks and dispatches to domain-specific worker agents.
2. Worker Agents: Specialized agents (Researcher Agent, Coder Agent, Quality Reviewer Agent).
3. Handoff Protocol: Shared state passing between supervisor and worker agents.
4. Visual Architecture: ASCII flow graph and Mermaid.js diagram.
"""

import sys
from typing import Any, Dict, List, Literal, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class MultiAgentState(TypedDict):
    user_request: str
    active_agent: str
    research_notes: str
    code_implementation: str
    review_comments: str
    final_output: str
    status: str


class SupervisorRouter:
    """Supervisor node deciding which worker agent should execute next."""
    
    def route(self, state: MultiAgentState) -> Literal["researcher", "coder", "reviewer", "synthesizer"]:
        request = state["user_request"].lower()
        
        if not state.get("research_notes"):
            print("[Supervisor Node] Decision: Dispatch task to -> [Researcher Agent]")
            return "researcher"
        elif not state.get("code_implementation"):
            print("[Supervisor Node] Decision: Dispatch task to -> [Coder Agent]")
            return "coder"
        elif not state.get("review_comments"):
            print("[Supervisor Node] Decision: Dispatch task to -> [Reviewer Agent]")
            return "reviewer"
        else:
            print("[Supervisor Node] Decision: All sub-tasks completed. Dispatch to -> [Synthesizer Node]")
            return "synthesizer"


# --- Sub-Agent Implementations ---

def researcher_agent(state: MultiAgentState) -> Dict[str, str]:
    print("  -> [Worker: Researcher Agent] Gathering architectural requirements and facts...")
    notes = f"Research findings for '{state['user_request']}': LangGraph supports cyclic graphs; Antigravity SDK supports A2A protocol."
    return {"research_notes": notes, "active_agent": "supervisor"}


def coder_agent(state: MultiAgentState) -> Dict[str, str]:
    print("  -> [Worker: Coder Agent] Writing production-ready implementation code...")
    code = (
        "class RouterAgent:\n"
        "    def route(self, input_data):\n"
        "        return 'dispatch_successful'"
    )
    return {"code_implementation": code, "active_agent": "supervisor"}


def reviewer_agent(state: MultiAgentState) -> Dict[str, str]:
    print("  -> [Worker: Reviewer Agent] Performing code audit & security check...")
    review = "Code audit passed. Clean class abstraction and clear interface."
    return {"review_comments": review, "active_agent": "supervisor"}


def synthesizer_node(state: MultiAgentState) -> Dict[str, str]:
    print("  -> [Synthesizer Node] Compiling final multi-agent deliverable...")
    final_text = (
        f"--- DELIVERABLE FOR: {state['user_request']} ---\n"
        f"1. Research: {state['research_notes']}\n"
        f"2. Implementation:\n{state['code_implementation']}\n"
        f"3. Quality Audit: {state['review_comments']}"
    )
    return {"final_output": final_text, "status": "completed"}


class MultiAgentOrchestrator:
    def __init__(self):
        self.supervisor = SupervisorRouter()

    def execute(self, user_request: str) -> MultiAgentState:
        state: MultiAgentState = {
            "user_request": user_request,
            "active_agent": "supervisor",
            "research_notes": "",
            "code_implementation": "",
            "review_comments": "",
            "final_output": "",
            "status": "in_progress"
        }
        
        print(f"\n[Multi-Agent System Started] Goal: '{user_request}'\n")

        while state["status"] != "completed":
            next_worker = self.supervisor.route(state)
            
            if next_worker == "researcher":
                update = researcher_agent(state)
                state.update(update)
            elif next_worker == "coder":
                update = coder_agent(state)
                state.update(update)
            elif next_worker == "reviewer":
                update = reviewer_agent(state)
                state.update(update)
            elif next_worker == "synthesizer":
                update = synthesizer_node(state)
                state.update(update)
                break
                
        print("\n[Multi-Agent Execution Finished] Deliverable compiled successfully.\n")
        return state


def render_mermaid_supervisor() -> str:
    return """```mermaid
graph TD
    User([User Prompt]) --> Supervisor[Supervisor Router Agent]
    
    Supervisor -->|Task 1: Research| Researcher[Researcher Sub-Agent]
    Supervisor -->|Task 2: Code| Coder[Coder Sub-Agent]
    Supervisor -->|Task 3: Audit| Reviewer[Reviewer Sub-Agent]
    
    Researcher -->|Return Notes| Supervisor
    Coder -->|Return Code| Supervisor
    Reviewer -->|Return Audit| Supervisor
    
    Supervisor -->|All Complete| Synthesizer[Synthesizer Node]
    Synthesizer --> Output([Final Deliverable])
    
    style User fill:#e1f5fe,stroke:#0288d1
    style Supervisor fill:#fff3e0,stroke:#f57c00
    style Researcher fill:#e8f5e9,stroke:#388e3c
    style Coder fill:#e8f5e9,stroke:#388e3c
    style Reviewer fill:#e8f5e9,stroke:#388e3c
    style Synthesizer fill:#f3e5f5,stroke:#7b1fa2
```"""


def render_ascii_supervisor():
    print("=" * 60)
    print("      MULTI-AGENT SUPERVISOR / ROUTER VISUALIZATION")
    print("=" * 60)
    print("""
                       +-------------------+
                       |    User Prompt    |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | Supervisor Router | <-------+
                       +----+----+----+----+         |
                            |    |    |              | State
            +---------------+    |    +-------+      | Updates
            |                    v            |      |
            v            +---------------+    v      |
    +---------------+    |  Coder Agent  | +-------+-+-------+
    | Researcher    |    +---------------+ | Reviewer Agent  |
    | Agent         |                      +-----------------+
    +---------------+
            ^                    ^                    ^
            |                    |                    |
            +--------------------+--------------------+
                                 |
                                 v (All Complete)
                       +-------------------+
                       | Synthesizer Node  |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | Final Response    |
                       +-------------------+
    """)
    print("=" * 60)


if __name__ == "__main__":
    render_ascii_supervisor()
    
    orchestrator = MultiAgentOrchestrator()
    final_state = orchestrator.execute("Design and implement a dynamic search agent system")
    
    print("[Final Deliverable Preview]:")
    print(final_state["final_output"])
    
    print("\n[Mermaid Multi-Agent Diagram]:")
    print(render_mermaid_supervisor())
