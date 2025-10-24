from langchain_core.tools import BaseTool
import logging
from pydantic import BaseModel, Field
from typing import Literal, List
import os
from common.stores.ClientStore import ClientStore
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentRegistrationInput(BaseModel):
    """Input schema for the RegisterAgent tool."""
    agent_id: str = Field(
        description="The agent id."
    )
    description: str = Field(
        description="A detailed natural language description of the agent's functions and specialities"
    )
    capabilities: List[str] = Field(
        description="A list of the agents capabilities."
    )


class RegisterAgent(BaseTool):
    """Tool class that inherits from base tool"""
    name: str = "RegisterAgent"
    description: str = "A tool that allows you to save agent information in a database"
    args_schema: type[BaseModel] = AgentRegistrationInput
    def _run(self,
             agent_id: str,
             description: str,
             capabilities: List[str],
             ) -> str:
        client = ClientStore()
        data = {
            "agent_id": agent_id,
            "description": description,
            "capabilities": capabilities,
            "status": "AVAILABLE"
        }
        result = client.store(data=data)
        logging.info("register tool called")
        logging.info(result)
        if result is None:
            return "Failed to save agent information"
        else:
            return "Agent information saved successfully"

