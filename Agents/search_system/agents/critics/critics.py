from google.adk.agents
from pydantic import BaseModel, Field
from typing import List

class Critique(BaseModel):
    """A structured result type of the critic agent"""
    issues: List[str] = Field(description="Concrete problems found in the draft")
    missing_evidence: List[str] = Field(description="Claims that lack support")
    suggested_revisions: List[str] = Field(description="Suggestions for improvement")

critics_agent = LlmAgent(
    name="critics",
    model="gemini-2.0-flash",
    description="Critically evaluates the Planner's draft answer.",
    instruction=(
        "Review the draft answer critically. Identify factual gaps, weak reasoning, "
        "or unsupported claims. Be specific — point to exact sentences, not vague "
        "impressions — and propose concrete revisions."
    ),
    output_schema=Critique,
    tools=[],
)
