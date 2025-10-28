"""
state structure to persist through conversations
"""

from typing import Annotated, Optional, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SQL2PDFAgentState(TypedDict):
    
    #convo history
    messages: Annotated[list, add_messages]

    # tool call cache
    execute_visualization_tool: Optional[str]   
    generate_pdf_report: Optional[str]  
    execute_sql_query: Optional[str]     

    # plan and reason
    plan: Optional[str]
    user_intent: Optional[str]

    # critique for self-review loop
    critique: Optional[str]

    # long term memory
    retrieved_memories: Optional[List[dict]]
    memory_context: Optional[str]

#empty  init state for new chat
def create_initial_state() -> SQL2PDFAgentState:
   
    return SQL2PDFAgentState(
        messages=[],
        execute_visualization_tool=None,
        generate_pdf_report=None,
        execute_sql_query=None,
        plan=None,
        user_intent=None,
        critique=None
    )
