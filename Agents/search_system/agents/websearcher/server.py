from google.adk.a2a.utils.agent_to_a2a import to_a2a

from agents.websearcher.websearcher import websearcher_agent

# Wraps the agent as an A2A-compatible Starlette app and auto-generates
# its agent card from the agent's name/description/instructions.
a2a_app = to_a2a(websearcher_agent, port=8101)
