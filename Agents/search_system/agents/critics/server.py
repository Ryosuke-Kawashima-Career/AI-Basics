from google.adk.a2a.utils.agent_to_a2a import to_a2a
from agents.critics.critics import critics_agent

a2a_app = to_a2a(critics_agent, port=8103)
