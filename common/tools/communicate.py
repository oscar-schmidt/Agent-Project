from langchain_core.tools import BaseTool
import asyncio
import logging
from pydantic import BaseModel, Field, create_model, PrivateAttr
from enum import Enum
from common.ConnectionManager import ConnectionManager
import json
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')





class Communicate(BaseTool):
    """Tool class that inherits from base tool"""
    name: str = "CommunicateTool"
    description: str = "A tool that allows you to contact other agents given a message, sender and recipient"
    args_schema: type[BaseModel]

    web_connection: ConnectionManager = Field(exclude=True)
    sender_id: str = Field(exclude=True)
    def _run(self, recipient_id: str, message: str):
        pass
        #return self._arun(sender_id, recipient_id, message)

    async def _arun(self, recipient_id: str, message: str):
        logging.info(f"ARUN id: {recipient_id}")
        logging.info(f"Web connection id: {self.web_connection.agent_id}")

        message_to_send = {
                    'message_type': 'message',
                    'recipient_id': recipient_id,
                    'sender_id': self.sender_id,
                    'message': message
                    }

        try:
            logging.info(f"Sending message: {message_to_send}")
            logging.info(f"[AGENT COMMUNICATION] Sender: {self.sender_id} -> Recipient: {recipient_id} | Message: {message[:150]}...")
            await self.web_connection.send(message_to_send)
            return "message sent successfully please wait fo the other agent to reply"
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return



def create_comm_tool(id:str, connection: ConnectionManager) -> Communicate:

    if id != "DirectoryAgent":
        CommunicateInput = create_model(
            f"{id} CommunicateInput",
            message = (str, Field(description="The content of the message to send.")),
            recipient_id = (str, Field(description="The recipient ID of the agent to contact, if you don't have information on the agent id of any other agent, "
                                                   "you may contact 'DirectoryAgent' which can help with finding other agents in the network")                        ),
        )
    else:
        CommunicateInput = create_model(
            f"{id} CommunicateInput",
            message=(str, Field(description="The content of the message to send.")),
            recipient_id=(str, Field(
                description="The recipient ID of the agent to contact, if you don't have information on the agent id you can use your tools")),

        )

    tool = Communicate(
        args_schema=CommunicateInput,
        web_connection=connection,
        sender_id=id,
    )
    return tool

