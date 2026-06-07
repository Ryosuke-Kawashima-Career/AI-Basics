from google.adk.agents import LlmAgent
from agents.tools import web_search_tool

websearcher_agent = LlmAgent(
    name="websearcher",
    model="gemini-2.0-flash",
    description="Gathers background information and sources for a research question.",
    instruction=(
        "You are a research assistant. Given a question, use the search tool to "
        "find relevant, current information. Summarize findings in 3-6 concise "
        "bullet points and always include the source URL for each claim."
    ),
    tools=[web_search_tool],
)