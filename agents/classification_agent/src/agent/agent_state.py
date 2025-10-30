"""
state structure to persist through conversations
"""

from typing import Annotated, Optional, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ReviewAgentState(TypedDict):

    #convo history
    messages: Annotated[list, add_messages]

    # tool call cache
    last_classified_reviews: Optional[str]
    last_sentiment_analysis: Optional[str]
    last_review_ids: Optional[List[str]]

    # plan and reason
    plan: Optional[str]
    user_intent: Optional[str]

    # critique for self-review loop
    critique: Optional[str]

    # long term memory
    retrieved_memories: Optional[List[dict]]
    memory_context: Optional[str]

    # workflow tracking
    workflow_steps_completed: Optional[List[str]]
    requires_agent_reply: Optional[bool]

#empty  init state for new chat
def create_initial_state() -> ReviewAgentState:
   
    return ReviewAgentState(
        messages=[],
        last_classified_reviews=None,
        last_sentiment_analysis=None,
        last_review_ids=None,
        plan=None,
        user_intent=None,
        critique=None
    )
