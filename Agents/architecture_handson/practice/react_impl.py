import json
import sys
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field

# 1. Mock tool functions
def search_database(query: str) -> str:
    """Mock search database tool."""
    if "LangGraph" in query:
        return "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph structures."
    return "No relevant information found."

def calculator(expression: str) -> str:
    """Mock calculator tool."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"

# 2. Tool Registry
class ToolRegistry(BaseModel):
    tools: Dict[str, Callable] = Field(default_factory=dict)
    
    def register(self, name: str, func: Callable):
        self.tools[name] = func
    
    def execute(self, name: str, *args, **kwargs) -> str:
        if name not in self.tools:
            return f"Tool '{name}' not found. Available tools: {list(self.tools.keys())}"
        try:
            return str(self.tools[name](*args, **kwargs))
        except Exception as e:
            return f"Error executing '{name}': {e}"

# 3. Mock ReAct LLM simulating reasoning steps
class ReActLLM(BaseModel):
    step_count: int = 0

    def invoke(self, user_query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        self.step_count += 1
        if self.step_count == 1:
            return {
                "thought": f"I need to search for information about: {user_query}",
                "action": "search_database",
                "action_input": user_query,
                "is_final_answer": False
            }
        if self.step_count == 2:
            return {
                "thought": "I have retrieved the search results and can now formulate the response.",
                "action": "Finish",
                "action_input": "LangGraph is a framework for building stateful, multi-actor LLM applications using cyclic graphs.",
                "is_final_answer": True
            }
        return {
            "thought": "All steps finished.",
            "action": "Finish",
            "action_input": "Finished processing.",
            "is_final_answer": True
        }

# 4. ReAct Agent Engine
class ReActAgent(BaseModel):
    llm: ReActLLM = Field(default_factory=ReActLLM)
    tools: ToolRegistry = Field(default_factory=ToolRegistry)
    history: List[Dict[str, str]] = Field(default_factory=list)

    def invoke(self, user_query: str, max_steps: int = 5) -> Dict[str, Any]:
        print(f"--- Agent Activated for Query: '{user_query}' ---\n")
        self.history.append({
            "role": "user",
            "content": user_query
        })
        for step in range(max_steps):
            print(f"Step {step + 1}:")
            result = self.llm.invoke(user_query, self.history)
            print(f"  Thought: {result.get('thought')}")
            print(f"  Action: {result.get('action')}")
            print(f"  Action Input: {result.get('action_input')}")

            if result.get("is_final_answer") or result.get("action") == "Finish":
                print(f"\nFinal Answer: {result.get('action_input')}")
                return result

            tool_name = result.get("action")
            tool_input = result.get("action_input")
            tool_output = self.tools.execute(tool_name, tool_input)

            print(f"  Observation: {tool_output}\n")
            self.history.append({
                "role": "agent",
                "thought": result.get("thought"),
                "action": tool_name,
                "observation": tool_output
            })
        return {"error": f"Max iterations ({max_steps}) reached.", "history": self.history}

if __name__ == "__main__":    
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    tools.register("calculator", calculator)
    
    agent = ReActAgent(llm=ReActLLM(), tools=tools)
    result = agent.invoke("What is LangGraph and how does it work?")

