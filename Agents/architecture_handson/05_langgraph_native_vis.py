"""
05_langgraph_native_vis.py
===========================
Architecture Pattern: Native LangGraph StateGraph & Built-in Graph Visualizer

Key Concepts:
1. Native LangGraph imports (`from langgraph.graph import StateGraph, START, END`).
2. State Schema using TypedDict.
3. Nodes, Edges, and Conditional Edges (`builder.add_conditional_edges`).
4. Built-in Graph Visualization (`graph.get_graph().draw_ascii()`).
"""

import sys
from typing import Dict, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 1. Define State Schema
class WorkflowState(TypedDict):
    input_text: str
    processed_text: str
    quality_score: int
    attempt: int


# 2. Node Functions
def process_node(state: WorkflowState) -> Dict[str, str]:
    print(f"[Node: Process] Processing input: '{state['input_text']}'...")
    return {"processed_text": f"PROCESSED: {state['input_text']}"}


def evaluate_node(state: WorkflowState) -> Dict[str, int]:
    attempt = state.get("attempt", 0) + 1
    # Increase score on second attempt to demonstrate cyclic loop
    score = 85 if attempt >= 2 else 60
    print(f"[Node: Evaluate] Attempt {attempt} -> Quality Score: {score}/100")
    return {"quality_score": score, "attempt": attempt}


# 3. Conditional Router Function
def router_condition(state: WorkflowState) -> Literal["process", "__end__"]:
    score = state.get("quality_score", 0)
    if score >= 80:
        print("  -> Conditional Router: Score >= 80. Routing to END.")
        return END
    else:
        print("  -> Conditional Router: Score < 80. Retrying Process Node.")
        return "process"


# 4. Build Native LangGraph
def build_native_langgraph():
    builder = StateGraph(WorkflowState)

    # Add Nodes
    builder.add_node("process", process_node)
    builder.add_node("evaluate", evaluate_node)

    # Add Edges
    builder.add_edge(START, "process")
    builder.add_edge("process", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        router_condition,
        {
            "process": "process",
            END: END
        }
    )

    return builder.compile()


if __name__ == "__main__":
    app = build_native_langgraph()
    
    print("=" * 60)
    print("       NATIVE LANGGRAPH BUILT-IN ASCII VISUALIZATION")
    print("=" * 60)
    try:
        ascii_graph = app.get_graph().draw_ascii()
        print(ascii_graph)
    except Exception as e:
        print(f"ASCII Graph render notice: {e}")
    print("=" * 60)
    
    print("\n[Executing Native LangGraph Workflow]:")
    initial_input: WorkflowState = {
        "input_text": "AI Agent Architecture Hands-on",
        "processed_text": "",
        "quality_score": 0,
        "attempt": 0
    }
    
    result = app.invoke(initial_input)
    
    print("\n[Execution Completed] Final State:")
    for k, v in result.items():
        print(f"  - {k}: {v}")
        
    print("\n[Native Mermaid Diagram Export]:")
    print("```mermaid")
    print(app.get_graph().draw_mermaid())
    print("```")
