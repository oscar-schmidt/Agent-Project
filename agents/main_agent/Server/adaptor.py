from asyncio import Queue
import asyncio
import json
import logging
from typing import Optional
from agents.main_agent.Server.ConnectionManager import ConnectionManager
from agents.main_agent.backend.model.states.tool_state.ToolReturnClass import ToolReturnClass


class Adaptor:
    _instance: Optional["Adaptor"] = None
    _lock = asyncio.Lock()

    def __init__(self, agent_id: str, description: str, capabilities: list[str]):
        self.connection = ConnectionManager(
            agent_id, description, capabilities)
        self.agent_id = agent_id
        self.message_queue = asyncio.Queue()
        self._started = False

    @classmethod
    async def get_adaptor(cls, agent_id, description, capabilities):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(agent_id, description, capabilities)
        return cls._instance


async def start(self, handler):
    logging.info(f"[{self.agent_id}] start() called with handler={handler}")

    if self._started:
        logging.info(f"[{self.agent_id}] Already started, skipping")
        return

    await self.connection.connect()
    logging.info(f"[{self.agent_id}] Connected to WebSocket server")

    asyncio.create_task(self.connection.start_listening(handler))
    logging.info(f"[{self.agent_id}] Listening task started in background")

    self._started = True
