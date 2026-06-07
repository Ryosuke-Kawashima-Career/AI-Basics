from google.adk.agents import LlmAgent

planner_agent = LlmAgent(
    name="planner",
    model="gemini-2.0-flash",
    instruction=(
        "You are drafting a first-pass answer to the user's query. If background "
        "research is provided, ground your answer in it and note where you relied "
        "on it. Be direct and structured — this draft will be critiqued, so it's "
        "fine to be imperfect, but make your reasoning explicit."
    ),
    tools=[],
)