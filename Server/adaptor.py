from asyncio import Queue
import logging

from Server.ConnectionManager import ConnectionManager
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass


class Adaptor:
    def __init__(self, agent_id: str, description: str, capabilities: list[str]):
        self.connection = ConnectionManager(
            agent_id, description, capabilities)
        self.state = None
        self.bind_tools = None
        self.message_queue: Queue[ToolReturnClass] = Queue()

    async def start(self):
        await self.connection.connect()
        await self.connection.start_listening(self.handle_message)

    async def handle_message(self, message_data):
        msg = message_data.get("message")
        sender_id = message_data.get("sender_id")

        tool_return_class = ToolReturnClass(
            state=self.state,
            agent_response=msg,
            meta={"tool_name": sender_id},
        )
        await self.message_queue.put(tool_return_class)

    async def send_message(self, payload: dict):
        await self.connection.send(payload)

    async def receive_message(self) -> ToolReturnClass:
        return await self.message_queue.get()
