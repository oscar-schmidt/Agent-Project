from langchain_core.tools import BaseTool
from enum import Enum
from pydantic import BaseModel, create_model, Field
from common.stores.ClientStore import ClientStore
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class Status(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class AgentUpdateInput(BaseModel):
    """Input schema for the update agent tool"""

    agent_id: str = Field(
        description="The ID of the agent that needs to be updated."
    )
    status: Status = Field(
        description="The new status for the agent."
    )
    
class UpdateAgentStatus(BaseTool):
    """
    Tool that updates when an agent becomes unavailable or available
    """
    name: str = "UpdateAgentStatus"
    description: str = "A tool that allows you to change the status of an agent"
    args_schema: type[BaseModel] = AgentUpdateInput
    
    def _run(self, agent_id: str, status: Status) -> dict | str | None:
        client = ClientStore()
        logging.info(f"Agent {agent_id} status update: {status}")
        result = client.update(agent_id=agent_id,status=status.value)
        logging.info(f"{result}")
        return "Update successful"


