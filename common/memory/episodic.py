from pydantic import BaseModel, Field


class Episode(BaseModel):
    """Write the episode from the perspetive of the agent"""

    observation: str = Field(description="The context and setup - what happend")

    thoughts: str = Field(description="Internal reasoning and process and observation of the agent")

    action: str = Field(description="What was done and in what format.")

    result: str = Field(description="Outcome and retrospective")


