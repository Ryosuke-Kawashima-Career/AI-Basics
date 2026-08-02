import os
import sys
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define Tools with @tool decorator
@tool
def search_database(query: str) -> str:
    """Search the database for information about a topic or framework."""
    if "LangGraph" in query or "langgraph" in query.lower():
        return "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph structures."
    return "No relevant information found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"

tools = [search_database, calculator]

# 2. Define the State of the System
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. LLM Setup with Tool Binding
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 4. LLM Node Function
def call_llm_node(state: State) -> dict:
    """Agent node that passes history to LLM with tools bound."""
    response = llm_with_tools.invoke(state["messages"])
    # Return dictionary with key 'messages'; add_messages reducer handles appending
    return {"messages": [response]}

# 5. Build StateGraph Workflow
def main():
    builder = StateGraph(State)

    # Add Nodes for Agent and Tools
    builder.add_node("agent", call_llm_node)
    builder.add_node("tools", ToolNode(tools))

    # Define Graph Control Flow (Edges)
    builder.add_edge(START, "agent")
    
    # Conditional edge out of 'agent':
    # If the LLM generates tool_calls -> route to 'tools' node; otherwise route to END
    builder.add_conditional_edges("agent", tools_condition)
    
    # Cyclic loop: return output from tool execution back to the agent node
    builder.add_edge("tools", "agent")

    app = builder.compile()

    # Invoke graph with user input
    query = "What is LangGraph and how does it work?"
    print(f"--- LangGraph Execution for: '{query}' ---\n")
    
    initial_input = {"messages": [HumanMessage(content=query)]}
    
    for chunk in app.stream(initial_input):
        for node_name, state_update in chunk.items():
            print(f"Node Executed: [{node_name}]")
            for msg in state_update.get("messages", []):
                msg.pretty_print()
            print()
            
    print("--- Graph Topology ---")
    try:
        print(app.get_graph().draw_ascii())
    except ImportError:
        print("(Note: Install 'grandalf' via `uv add grandalf` to render ASCII graph diagrams)")

if __name__ == '__main__':
    main()
