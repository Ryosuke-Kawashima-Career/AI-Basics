"""
04_plan_and_execute_vis.py
===========================
Architecture Pattern: Plan-and-Execute Agent Architecture

Key Concepts:
1. High-Level Planner: Deconstructs a user goal into discrete, ordered tasks.
2. Step Executor: Focuses on executing a single task step using tools.
3. Dynamic Re-Planner: Evaluates step execution results, adjusts remaining steps, or terminates.
4. Visual Architecture: ASCII flow graph and Mermaid.js sequence diagram.
"""

import sys
from typing import Any, Dict, List, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class PlanExecuteState(TypedDict):
    objective: str
    plan: List[str]
    completed_steps: List[str]
    results: Dict[str, str]
    final_response: str
    status: str


def planner_node(state: PlanExecuteState) -> Dict[str, Any]:
    print(f"[Planner Node] Deconstructing objective: '{state['objective']}' into actionable steps...")
    plan = [
        "Search current documentation for AI Agent patterns",
        "Generate comparison matrix",
        "Synthesize recommendation"
    ]
    print(f"  -> Generated Plan ({len(plan)} steps): {plan}")
    return {"plan": plan, "completed_steps": [], "results": {}}


def step_executor_node(state: PlanExecuteState) -> Dict[str, Any]:
    current_step = state["plan"][0]
    print(f"  -> [Step Executor] Executing active step: '{current_step}'...")
    
    # Mock tool execution based on step
    if "Search" in current_step:
        output = "Found 4 primary patterns: ReAct, StateGraph, Supervisor, Plan-and-Execute."
    elif "matrix" in current_step:
        output = "Matrix created: LangGraph excels at state loops; Antigravity SDK excels at A2A supervisor handoffs."
    else:
        output = "Synthesis complete."
        
    updated_completed = list(state["completed_steps"]) + [current_step]
    updated_results = dict(state["results"])
    updated_results[current_step] = output
    remaining_plan = state["plan"][1:]
    
    return {
        "plan": remaining_plan,
        "completed_steps": updated_completed,
        "results": updated_results
    }


def replanner_node(state: PlanExecuteState) -> Dict[str, Any]:
    remaining = state["plan"]
    completed = state["completed_steps"]
    print(f"[Re-Planner Node] Assessing progress ({len(completed)} completed, {len(remaining)} remaining)...")
    
    if not remaining:
        print("  -> Plan fully executed! Synthesizing final answer...")
        summary = "\n".join([f"- Step '{k}': {v}" for k, v in state["results"].items()])
        final_ans = f"Objective Accomplished:\n{summary}"
        return {"final_response": final_ans, "status": "completed"}
    else:
        print(f"  -> Next step to execute: '{remaining[0]}'")
        return {"status": "in_progress"}


class PlanAndExecuteAgent:
    def run(self, objective: str) -> PlanExecuteState:
        state: PlanExecuteState = {
            "objective": objective,
            "plan": [],
            "completed_steps": [],
            "results": {},
            "final_response": "",
            "status": "initialized"
        }
        
        print(f"\n[Plan & Execute Agent Started] Objective: '{objective}'\n")
        
        # Step 1: Initial Planning
        update = planner_node(state)
        state.update(update)
        
        # Execution & Re-planning Loop
        while state["status"] != "completed":
            update = step_executor_node(state)
            state.update(update)
            
            update = replanner_node(state)
            state.update(update)
            
        print("\n[Plan & Execute Execution Finished]\n")
        return state


def render_mermaid_plan_execute() -> str:
    return """```mermaid
graph TD
    User([User Goal]) --> Planner[Planner Node]
    Planner -->|Initial Plan| Executor[Step Executor Node]
    Executor -->|Step Result| Replanner[Re-Planner Node]
    
    Replanner -->|Steps Remaining?| Executor
    Replanner -->|Plan Complete| Synthesis[Final Synthesis]
    Synthesis --> Output([User Response])
    
    style User fill:#e1f5fe,stroke:#0288d1
    style Planner fill:#fff3e0,stroke:#f57c00
    style Executor fill:#e8f5e9,stroke:#388e3c
    style Replanner fill:#ffe0b2,stroke:#e65100
    style Synthesis fill:#f3e5f5,stroke:#7b1fa2
```"""


def render_ascii_plan_execute():
    print("=" * 60)
    print("        PLAN & EXECUTE AGENT VISUALIZATION")
    print("=" * 60)
    print("""
                       +-------------------+
                       |     User Goal     |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       |   Planner Node    |
                       +---------+---------+
                                 |
                                 v
          +--------------> +-------------------+
          |                |   Step Executor   |
          |                +---------+---------+
          |                          |
     Steps Remaining                 v
          |                +-------------------+
          +----------------|  Re-Planner Node  |
                           +---------+---------+
                                     |
                               All Completed
                                     v
                           +-------------------+
                           |  Final Synthesis  |
                           +-------------------+
    """)
    print("=" * 60)


if __name__ == "__main__":
    render_ascii_plan_execute()
    
    agent = PlanAndExecuteAgent()
    final_state = agent.run("Compare modern AI Agent Architecture patterns")
    
    print("[Final Output Summary]:")
    print(final_state["final_response"])
    
    print("\n[Mermaid Plan & Execute Diagram]:")
    print(render_mermaid_plan_execute())
