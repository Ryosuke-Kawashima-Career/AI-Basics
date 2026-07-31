"""
01_react_agent_vis.py
======================
Architecture Pattern: ReAct (Reasoning + Acting) Single-Agent Loop

Key Concepts:
1. Thought: LLM decides what step to take next based on past observations.
2. Action: LLM selects a tool and provides arguments.
3. Observation: Tool executes and returns output back into the message history.
4. Visualization: Generates ASCII architecture tree and Mermaid.js diagram.
"""

import json
import sys
from typing import Any, Callable, Dict, List

# Reconfigure encoding for safe Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._tools[name] = func

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."
        try:
            return str(self._tools[name](**kwargs))
        except Exception as e:
            return f"Error executing '{name}': {str(e)}"

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())


def search_database(query: str) -> str:
    """Mock search tool returning database info."""
    data = {
        "AI Agent": "An autonomous entity driven by LLMs, state memory, tools, and perception loops.",
        "LangGraph": "A library for building stateful, multi-actor applications with LLMs using graphs.",
        "Antigravity SDK": "Google SDK for building modular multi-agent systems and agent-to-agent (A2A) protocols."
    }
    for k, v in data.items():
        if k.lower() in query.lower():
            return f"Found match for '{k}': {v}"
    return f"No direct match found for query: '{query}'."


def calculator(expression: str) -> str:
    """Mock safe calculator tool."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression."
        return str(eval(expression))
    except Exception as e:
        return f"Calc error: {e}"


class MockReActLLM:
    """Simulates an LLM producing ReAct Thought + Action or Final Answer."""
    
    def __init__(self):
        self.step_count = 0

    def generate(self, user_query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        self.step_count += 1
        
        # Step 1: Decide to search
        if self.step_count == 1:
            return {
                "thought": "I need to search for information about AI Agent architecture and LangGraph.",
                "action": "search_database",
                "action_input": {"query": "LangGraph"},
                "is_final": False
            }
        # Step 2: Return final answer
        elif self.step_count == 2:
            return {
                "thought": "I have found information on LangGraph. Now I will provide a final synthesis.",
                "action": None,
                "action_input": None,
                "final_answer": "LangGraph is a framework designed for stateful, multi-actor LLM applications using cyclic graphs.",
                "is_final": True
            }
        return {"thought": "Done", "is_final": True, "final_answer": "Task complete."}


class ReActAgent:
    def __init__(self, llm: MockReActLLM, tools: ToolRegistry):
        self.llm = llm
        self.tools = tools
        self.history: List[Dict[str, str]] = []

    def run(self, user_query: str, max_steps: int = 5) -> str:
        print(f"\n[Agent Initialized] User Query: '{user_query}'\n")
        self.history.append({"role": "user", "content": user_query})

        for i in range(max_steps):
            response = self.llm.generate(user_query, self.history)
            
            print(f"[Step {i+1}] Thought: {response['thought']}")
            
            if response.get("is_final"):
                print(f"\n[Final Answer]: {response.get('final_answer')}\n")
                return response.get("final_answer", "")
            
            tool_name = response["action"]
            tool_input = response["action_input"]
            print(f"[Action]: Execute '{tool_name}' with args {tool_input}")
            
            observation = self.tools.execute(tool_name, **tool_input)
            print(f"[Observation]: {observation}\n")
            
            self.history.append({
                "role": "assistant",
                "thought": response["thought"],
                "action": tool_name,
                "observation": observation
            })
            
        return "Max iterations reached."


def generate_mermaid_diagram() -> str:
    return """```mermaid
graph TD
    User([User Query]) --> LLMNode[LLM Reasoner]
    LLMNode -->|Decides Tool| ActionNode[Tool Call Execution]
    ActionNode -->|Returns Result| LLMNode
    LLMNode -->|Final Synthesis| Output([Final Response])
    
    style User fill:#e1f5fe,stroke:#0288d1
    style LLMNode fill:#fff3e0,stroke:#f57c00
    style ActionNode fill:#e8f5e9,stroke:#388e3c
    style Output fill:#f3e5f5,stroke:#7b1fa2
```"""


def render_ascii_architecture():
    print("=" * 60)
    print("           REACT AGENT ARCHITECTURE VISUALIZATION")
    print("=" * 60)
    print("""
       +--------------------+
       |     User Query     |
       +---------+----------+
                 |
                 v
        +------------------+ <-----+
        |  LLM Thought &   |       |
        |  Action Selector |       | Observation /
        +--------+---------+       | Feedback Loop
                 |                 |
     +-----------+-----------+     |
     |                       |     |
     v                       v     |
+---------+            +-----------+--+
|  Final  |            |  Tool Execution |
| Answer  |            | (Search, Calc)  |
+---------+            +--------------+--+
    """)
    print("=" * 60)


if __name__ == "__main__":
    render_ascii_architecture()
    
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    tools.register("calculator", calculator)
    
    agent = ReActAgent(llm=MockReActLLM(), tools=tools)
    agent.run("What is LangGraph and how does it work?")
    
    print("\n[Mermaid Diagram Specification]:")
    print(generate_mermaid_diagram())

