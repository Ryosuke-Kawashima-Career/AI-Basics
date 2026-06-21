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
    model="gemini-1.5-flash",
    description="Coordinates research, drafting, and critique to answer the user.",
    instruction=(
        "You are a research coordinator. Your goal is to answer the user's question thoroughly and accurately.\n\n"
        "Process:\n"
        "1. Use the websearcher agent to gather background information and sources.\n"
        "2. Use the planner agent to draft an initial answer based on the research.\n"
        "3. Use the critics agent to evaluate and improve the draft answer.\n"
        "4. Provide the final, refined answer to the user.\n\n"
        "Be thorough in your research, clear in your explanations, and open to constructive criticism."
    ),
    sub_agents=[websearcher_remote, planner_remote, critics_remote],
)
