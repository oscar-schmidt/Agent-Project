import time

from langchain_core.tools import BaseTool
import logging
from pydantic import BaseModel, create_model, Field
import os
from common.stores.ClientStore import ClientStore
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')



class AgentRegistrationInput(BaseModel):
    """Input schema for the RetrieveAgent tool."""

    description: str = Field(
        description="A detailed description of the sort of problem the agent needs help with"
    )
    
class RetrieveAgent(BaseTool):
    name: str = "RetrieveAgentInformation"
    description: str = "A tool that allows you to get agent information through semantic search, given a description"
    args_schema: type[BaseModel] = AgentRegistrationInput
    
    def _run(self, description: str) -> dict | str | None:
        logging.info(f"Running {description}")
        client = ClientStore()
        logging.info("retrieve called")
        data = {
            "description": description,
        }
        result = client.get(quary=data)
        logging.info(f"{result}")
        return result

