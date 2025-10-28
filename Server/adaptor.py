from asyncio import Queue
import asyncio
import json
import logging
from typing import Optional

from Server.ConnectionManager import ConnectionManager
from backend.model.states.tool_state.ToolReturnClass import ToolReturnClass


class Adaptor:
    _instance: Optional["Adaptor"] = None
    _lock = asyncio.Lock()

    def __init__(self, agent_id: str, description: str, capabilities: list[str]):
        self.connection = ConnectionManager(
            agent_id, description, capabilities)
        self.state = None
        self.bind_tools = None
        self.message_queue: Queue[ToolReturnClass] = Queue()

    @classmethod
    async def get_adaptor(cls, agent_id: str = None, description: str = None, capabilities: list[str] = None):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(agent_id, description, capabilities)

            # Access cls._instance directly instead of using a local variable
            if not cls._instance.connection or not cls._instance.connection.is_connected:
                await cls._instance.connection._reconnect()

        return cls._instance

    async def start(self):
        if getattr(self, "_started", False):
            return
        await self.connection.connect()
        asyncio.create_task(
            self.connection.start_listening(self.handle_message))
        self._started = True

    async def handle_message(self, message_data):
        try:
            msg = message_data.get("message") or message_data.get("content")
            sender_id = message_data.get("sender_id")
            msg_type = message_data.get(
                "message_type") or message_data.get("type")

            if msg_type in ["pong", "registered", "error"]:
                if msg_type == "error":
                    logging.error(
                        f"Server error: {message_data.get('message')}")
                return

            if msg and sender_id:
                tool_return_class = ToolReturnClass(
                    state=self.state,
                    agent_response=msg,
                    meta={"tool_name": sender_id},
                )
                await self.message_queue.put(tool_return_class)
                logging.info(f"Message from {sender_id} queued")
            else:
                logging.warning(
                    f"Received message with missing fields: {message_data}")

        except Exception as e:
            logging.error(f"Error handling message: {e}", exc_info=True)

    async def send_message(self, payload: dict):
        try:
            await self.connection.send(payload)
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            await self.connection._reconnect()

    async def receive_message(self) -> ToolReturnClass:
        try:
            return await self.message_queue.get()
        except Exception as e:
            logging.error(f"Error receiving message: {e}")
            await self.connection._reconnect()
