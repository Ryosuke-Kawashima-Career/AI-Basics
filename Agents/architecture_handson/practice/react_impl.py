import json
import sys
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field

class ToolRegistry(BaseModel):
    tools: Dict[str, Callable]
    
    def register(self, name: str, func: Callable):
        self.tools[name] = func
    
    def execute(self, name: str, *args, **kwargs) -> str:
        if name not in self.tools:
            return f"Tool {name} not found."
        try:
            return str(self.tools[name](*args, **kwargs))
        except Exception as e:
            return f"Error: {e}"

class ReActLLM(BaseModel):
    llm: Model
    step_count: int = 0

    def invoke(self, user_query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        self.step_count += 1
        if self.step_count == 1:
            return {
                "thought": f"This is the first step, I need to understand the user query: {user_query}",
                "action": "Search",
                "action_input": ""
            }
        if self.step_count == 2:
            return {
                "thought": "I need to use the Search tool to find the weather information.",
                "action": "Search",
                "action_input": "What is the weather like today?
            }
        return {
            "thought": "I have all the information I need, I can now answer the user query.",
            "action": "Finish",
            "action_input": "The weather is sunny today."
        }


class ReActAgent(BaseModel):
    llm: ReActLLM
    tools: ToolRegistry
    history: List[Dict[str, str]] = []

    def invoke(self, user_query: str, max_steps=5) -> str:
        print(f"The agent has been activated for {user_query}\n")
        self.history.append({
            "role": user_query,
            "content": user_query
        })
        for step in range(max_steps):
            print(f"Step {step + 1}")
            result = self.llm.invoke(user_query, self.history)
            if result.get("is_final_answer"):
                return result
            tool_name = result.get("action")
            tool_input = result.get("action_input")
            tool_output = self.tools.execute(tool_name, tool_input)

            print(f"Tool Output: {tool_output}\n")
            self.history.append({
                "role": "agent",
                "thought": result.get("thought"),
                "action": result.get("action"),
                "observation": tool_output
            })
        return "Max_iteration reached with the given processing steps " + str(self.history)

if __name__ == "__main__":    
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    tools.register("calculator", calculator)
    
    agent = ReActAgent(llm=ReActLLM(), tools=tools)
    result = agent.invoke("What is LangGraph and how does it work?")
    
    print(result)
