"""
02_langgraph_workflow_vis.py
=============================
Architecture Pattern: Stateful Cyclic Graph (LangGraph / State Graph Pattern)

Key Concepts:
1. Shared State: A TypedDict containing query, context, draft, review_score, and step counter.
2. Nodes: Functional units of computation (Planner Node -> Writer Node -> Evaluator Node).
3. Conditional Edges: Decisions after Evaluator Node (if score < 80 -> route back to Writer Node, else -> END).
4. Visual Graph Exporter: Renders ASCII state machine diagram and Mermaid graph.
"""

import sys
from typing import Any, Dict, List, Literal, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 1. Define Agent State Schema
class AgentState(TypedDict):
    query: str
    plan: str
    draft: str
    review_score: int
    review_feedback: str
    iteration: int
    status: str


# 2. Define Node Functions
def planner_node(state: AgentState) -> Dict[str, str]:
    print("[Node: Planner] Generating execution plan...")
    query = state["query"]
    plan = f"Plan for '{query}': 1. Explain concept 2. Provide code snippet 3. Summarize."
    return {"plan": plan, "status": "planned"}


def writer_node(state: AgentState) -> Dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    print(f"[Node: Writer] Generating content draft (Iteration {iteration})...")
    
    feedback = state.get("review_feedback", "")
    if iteration == 1:
        draft = "LangGraph is a library for stateful LLM applications."
    else:
        draft = f"LangGraph enables cyclic state graphs for LLMs. Addressed feedback: {feedback}"
        
    return {"draft": draft, "iteration": iteration, "status": "drafted"}


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    print("[Node: Evaluator] Reviewing content quality...")
    draft = state.get("draft", "")
    iteration = state.get("iteration", 1)
    
    # Simulate evaluation: pass after 2 iterations
    if iteration >= 2 or len(draft) > 50:
        score = 90
        feedback = "Comprehensive and clear description."
    else:
        score = 65
        feedback = "Draft is too brief. Please expand on cyclic state features."
        
    print(f"  -> Quality Score: {score}/100 | Feedback: '{feedback}'")
    return {"review_score": score, "review_feedback": feedback, "status": "evaluated"}


# 3. Router / Conditional Edge Function
def router_edge(state: AgentState) -> Literal["writer", "end"]:
    score = state.get("review_score", 0)
    iteration = state.get("iteration", 0)
    
    if score >= 80 or iteration >= 3:
        print("  -> Router Decision: Score target met or max retries reached. Routing to [END].")
        return "end"
    else:
        print("  -> Router Decision: Score below target. Routing back to [Writer Node].")
        return "writer"


# 4. State Machine Runner
class SimpleStateGraphRunner:
    def __init__(self):
        self.state: AgentState = {
            "query": "",
            "plan": "",
            "draft": "",
            "review_score": 0,
            "review_feedback": "",
            "iteration": 0,
            "status": "initialized"
        }

    def run(self, initial_query: str) -> AgentState:
        self.state["query"] = initial_query
        print(f"\n[Graph Execution Started] Input: '{initial_query}'\n")

        # Step 1: Planner
        update = planner_node(self.state)
        self.state.update(update)

        # Step 2: Writer Loop
        while True:
            update = writer_node(self.state)
            self.state.update(update)

            # Step 3: Evaluator
            update = evaluator_node(self.state)
            self.state.update(update)

            # Conditional Edge
            next_node = router_edge(self.state)
            if next_node == "end":
                break

        print(f"\n[Graph Execution Completed] Final Score: {self.state['review_score']}\n")
        return self.state


def render_mermaid_stategraph() -> str:
    return """```mermaid
graph TD
    START([__start__]) --> Planner[Planner Node]
    Planner --> Writer[Writer Node]
    Writer --> Evaluator[Evaluator Node]
    Evaluator -->|Score >= 80| END([__end__])
    Evaluator -->|Score < 80 & Iteration < 3| Writer
    
    style START fill:#c8e6c9,stroke:#2e7d32
    style Planner fill:#e1f5fe,stroke:#0288d1
    style Writer fill:#fff3e0,stroke:#f57c00
    style Evaluator fill:#ffe0b2,stroke:#e65100
    style END fill:#ffcdd2,stroke:#c62828
```"""


def render_ascii_stategraph():
    print("=" * 60)
    print("        LANGGRAPH CYCLIC STATE GRAPH VISUALIZATION")
    print("=" * 60)
    print("""
                   +----------------+
                   |   __START__    |
                   +-------+--------+
                           |
                           v
                   +----------------+
                   |  Planner Node  |
                   +-------+--------+
                           |
                           v
          +--------------> +----------------+
          |                |  Writer Node   |
          |                +-------+--------+
          |                        |
          |                        v
          |                +----------------+
          |                | Evaluator Node |
          |                +-------+--------+
          |                        |
     Score < 80?                   v
   (Refine Draft)      /------------------------\\
          +-----------| Conditional Edge Router  |
                      \\------------------------/
                                   |
                             Score >= 80?
                                   |
                                   v
                           +----------------+
                           |    __END__     |
                           +----------------+
    """)
    print("=" * 60)


if __name__ == "__main__":
    render_ascii_stategraph()
    runner = SimpleStateGraphRunner()
    final_state = runner.run("Explain LangGraph Architecture")
    
    print("[Final Shared State Snapshot]:")
    for k, v in final_state.items():
        print(f"  - {k}: {v}")

    print("\n[Mermaid Graph Specification]:")
    print(render_mermaid_stategraph())
