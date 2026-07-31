"""
quizzes.py
===========
Interactive AI Agent Architecture Coding Quizzes & Self-Assessment Suite

Run this file directly to test your architectural implementation:
    python Agents/architecture_handson/quizzes.py
"""

import sys
from typing import Any, Dict, List, Literal, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================
# QUIZ 1: STATE GRAPH CONDITIONAL ROUTING (LangGraph Pattern)
# =====================================================================

class Quiz1State(TypedDict):
    query: str
    error_count: int
    validation_passed: bool
    status: str


def quiz1_should_continue(state: Quiz1State) -> Literal["retry_node", "success_node", "escalate_node"]:
    """
    QUIZ 1 TASK:
    Implement the conditional routing logic for a self-healing agent state graph.
    
    Routing Rules:
    1. If `validation_passed` is True -> return "success_node".
    2. If `validation_passed` is False AND `error_count` < 3 -> return "retry_node".
    3. If `validation_passed` is False AND `error_count` >= 3 -> return "escalate_node".
    """
    # --- SOLUTION IMPLEMENTATION ---
    if state.get("validation_passed", False):
        return "success_node"
    elif state.get("error_count", 0) < 3:
        return "retry_node"
    else:
        return "escalate_node"


def run_quiz1_tests():
    print("--- [Testing Quiz 1: State Graph Router] ---")
    
    test_cases = [
        ({"query": "q1", "error_count": 0, "validation_passed": True, "status": "ok"}, "success_node"),
        ({"query": "q2", "error_count": 1, "validation_passed": False, "status": "err"}, "retry_node"),
        ({"query": "q3", "error_count": 3, "validation_passed": False, "status": "err"}, "escalate_node"),
    ]
    
    passed = 0
    for idx, (state, expected) in enumerate(test_cases, 1):
        res = quiz1_should_continue(state)
        if res == expected:
            print(f"  [Test {idx}] PASSED: State {state} -> Routed to '{res}'")
            passed += 1
        else:
            print(f"  [Test {idx}] FAILED: Expected '{expected}', got '{res}'")
            
    print(f"Quiz 1 Result: {passed}/{len(test_cases)} tests passed.\n")


# =====================================================================
# QUIZ 2: MULTI-AGENT SUPERVISOR DISPATCH & STATE MERGE
# =====================================================================

class Quiz2State(TypedDict):
    pending_tasks: List[str]
    completed_results: Dict[str, str]


def quiz2_supervisor_dispatch(state: Quiz2State) -> str:
    """
    QUIZ 2 TASK:
    Given a list of pending tasks in state['pending_tasks'], return the name of the next sub-agent to dispatch:
    - If task starts with 'search:' -> dispatch to "search_agent"
    - If task starts with 'code:' -> dispatch to "coding_agent"
    - If task list is empty -> return "finish"
    """
    # --- SOLUTION IMPLEMENTATION ---
    tasks = state.get("pending_tasks", [])
    if not tasks:
        return "finish"
    next_task = tasks[0]
    if next_task.startswith("search:"):
        return "search_agent"
    elif next_task.startswith("code:"):
        return "coding_agent"
    return "unknown_agent"


def run_quiz2_tests():
    print("--- [Testing Quiz 2: Multi-Agent Dispatch] ---")
    
    test_cases = [
        ({"pending_tasks": ["search:Find graph algorithms", "code:Implement BFS"], "completed_results": {}}, "search_agent"),
        ({"pending_tasks": ["code:Implement BFS"], "completed_results": {}}, "coding_agent"),
        ({"pending_tasks": [], "completed_results": {}}, "finish"),
    ]
    
    passed = 0
    for idx, (state, expected) in enumerate(test_cases, 1):
        res = quiz2_supervisor_dispatch(state)
        if res == expected:
            print(f"  [Test {idx}] PASSED: Dispatch -> '{res}'")
            passed += 1
        else:
            print(f"  [Test {idx}] FAILED: Expected '{expected}', got '{res}'")
            
    print(f"Quiz 2 Result: {passed}/{len(test_cases)} tests passed.\n")


# =====================================================================
# QUIZ 3: HUMAN-IN-THE-LOOP (HITL) APPROVAL GATE
# =====================================================================

class Quiz3State(TypedDict):
    proposed_action: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    user_approved: bool
    execution_status: str


def quiz3_human_approval_gate(state: Quiz3State) -> Dict[str, Any]:
    """
    QUIZ 3 TASK:
    Implement a Human-in-the-Loop (HITL) node for agent safety boundaries:
    - If risk_level is "HIGH" and user_approved is False:
        set execution_status to "PAUSED_PENDING_APPROVAL"
    - Otherwise (risk is LOW/MEDIUM or user_approved is True):
        set execution_status to "APPROVED_FOR_EXECUTION"
    """
    # --- SOLUTION IMPLEMENTATION ---
    risk = state.get("risk_level", "LOW")
    approved = state.get("user_approved", False)
    
    if risk == "HIGH" and not approved:
        return {"execution_status": "PAUSED_PENDING_APPROVAL"}
    else:
        return {"execution_status": "APPROVED_FOR_EXECUTION"}


def run_quiz3_tests():
    print("--- [Testing Quiz 3: Human-in-the-Loop Gate] ---")
    
    test_cases = [
        ({"proposed_action": "delete_db", "risk_level": "HIGH", "user_approved": False, "execution_status": ""}, "PAUSED_PENDING_APPROVAL"),
        ({"proposed_action": "delete_db", "risk_level": "HIGH", "user_approved": True, "execution_status": ""}, "APPROVED_FOR_EXECUTION"),
        ({"proposed_action": "read_logs", "risk_level": "LOW", "user_approved": False, "execution_status": ""}, "APPROVED_FOR_EXECUTION"),
    ]
    
    passed = 0
    for idx, (state, expected) in enumerate(test_cases, 1):
        res = quiz3_human_approval_gate(state)
        actual = res["execution_status"]
        if actual == expected:
            print(f"  [Test {idx}] PASSED: Risk '{state['risk_level']}' / Approved '{state['user_approved']}' -> Status '{actual}'")
            passed += 1
        else:
            print(f"  [Test {idx}] FAILED: Expected '{expected}', got '{actual}'")
            
    print(f"Quiz 3 Result: {passed}/{len(test_cases)} tests passed.\n")


def print_quiz_summary():
    print("=" * 60)
    print("        AI AGENT ARCHITECTURE HANDS-ON QUIZ SUITE")
    print("=" * 60)
    print("Run all verification tests...\n")
    run_quiz1_tests()
    run_quiz2_tests()
    run_quiz3_tests()
    print("=" * 60)
    print("All architectural quizzes verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    print_quiz_summary()
