from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH, RemoteA2aAgent

websearcher_remote = RemoteA2aAgent(
    name="websearcher",
    description="Gathers background information and sources for a research question.",
    agent_card=f"http://localhost:8101{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)
planner_remote = RemoteA2aAgent(
    name="planner",
    description="Drafts an initial answer to the user's query.",
    agent_card=f"http://localhost:8102{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)
critics_remote = RemoteA2aAgent(
    name="critics",
    description="Critically evaluates the Planner's draft answer.",
    agent_card=f"http://localhost:8103{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)
 
coordinator_agent = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash",
    description="Coordinates research, drafting, and critique to answer the user.",
    instruction="...",
    sub_agents=[websearcher_remote, planner_remote, critics_remote],
)
