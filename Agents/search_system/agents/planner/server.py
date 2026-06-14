from google.adk.a2a.utils.agent_to_a2a import to_a2a
from agents.planner.planner import planner_agent

a2a_app = to_a2a(planner_agent, port=8102)