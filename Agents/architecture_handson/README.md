# AI Agent Architecture Hands-on Guide

Welcome to the **AI Agent Architecture Hands-on** course module. This repository provides runnable Python implementations, visual architectural diagrams (ASCII & Mermaid.js), and interactive quizzes covering modern AI agent design patterns.

---

## 🎯 Architectural Patterns Covered

| Pattern | Description | Key Mechanism | Best Use Case |
| :--- | :--- | :--- | :--- |
| **1. ReAct (Reasoning + Acting)** | Single-agent execution loop interspersing thought, tool selection, and tool execution. | Loop: `LLM Thought -> Tool Call -> Environment -> Observation -> LLM` | Simple Q&A, single-domain search, basic API callers. |
| **2. Cyclic State Graph (LangGraph Pattern)** | Deterministic state machine where nodes update a shared state dictionary and conditional edges control flow. | Shared TypedDict State + Router Edge conditions | Complex multi-step reasoning, evaluators, retry loops. |
| **3. Supervisor / Router (Antigravity SDK / ADK)** | Hierarchical multi-agent architecture where a Supervisor LLM routes sub-tasks to specialized domain agents. | Handoff State + Specialized Agent registry | Enterprise applications, multi-domain problem solving. |
| **4. Plan & Execute Agent** | Decouples high-level task planning from execution, dynamically updating the plan based on tool feedback. | `Planner Node -> Step Executor Node -> Re-planner Node` | Long-horizon goals, coding tasks, research reports. |

---

## 🎨 Visualization Techniques

Visualization is critical for understanding agent control flow and debugging cycles. This hands-on module supports two visualization outputs:

1. **Terminal ASCII Visualizers**: Render node trees and state flow directly in stdout without external dependencies.
2. **Mermaid.js Markup**: Generate standard Mermaid diagram code (ready to render in GitHub, VS Code, or web viewers).

---

## 🚀 Quickstart & Directory Structure

```text
Agents/architecture_handson/
├── README.md                          # This guide
├── 01_react_agent_vis.py              # Single-Agent ReAct Architecture with visualization
├── 02_langgraph_workflow_vis.py       # LangGraph Cyclic Workflow Architecture
├── 03_multiagent_supervisor_vis.py    # Multi-Agent Supervisor / Router Architecture
├── 04_plan_and_execute_vis.py         # Plan-and-Execute Architecture
├── 05_langgraph_native_vis.py         # Native LangGraph StateGraph API & Built-in Visualizer
└── quizzes.py                         # Interactive Quizzes & Hands-on Coding Challenges
```

### Running the Examples

```bash
# Run ReAct Agent Architecture example
python Agents/architecture_handson/01_react_agent_vis.py

# Run LangGraph State Graph Workflow
python Agents/architecture_handson/02_langgraph_workflow_vis.py

# Run Multi-Agent Supervisor / Router
python Agents/architecture_handson/03_multiagent_supervisor_vis.py

# Run Plan-and-Execute Agent Architecture
python Agents/architecture_handson/04_plan_and_execute_vis.py

# Run Native LangGraph StateGraph & Built-in Visualizer
python Agents/architecture_handson/05_langgraph_native_vis.py

# Run Interactive Quizzes & Self-Assessment
python Agents/architecture_handson/quizzes.py
```

---

## 📝 Hands-on Quizzes & Challenges

Open [quizzes.py](file:///d:/Training/AI-Basics/Agents/architecture_handson/quizzes.py) to attempt the 3 hands-on coding challenges:
1. **Quiz 1: State Machine Conditional Routing**: Implement a `should_continue()` router function in a cyclic graph state.
2. **Quiz 2: Supervisor State Merge & Worker Handoff**: Construct a supervisor dispatch mechanism.
3. **Quiz 3: Human-in-the-Loop Node**: Design an approval boundary before critical tool execution.
