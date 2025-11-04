import asyncio
import json
import logging
from typing import Optional, Callable
from common.ConnectionManager import ConnectionManager


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

    async def start(self, handler: Callable):
        logging.info(f"[{self.agent_id}] start() called")

        if self._started:
            logging.info(f"[{self.agent_id}] Already started, skipping")
            return

        try:
            await self.connection.connect()
            logging.info(f"[{self.agent_id}] Connected to WebSocket server")

            await self.connection.start_listening(handler)
            logging.info(f"[{self.agent_id}] Listening task started")

            self._started = True

        except Exception as e:
            logging.error(
                f"[{self.agent_id}] Failed to start: {e}", exc_info=True)
            raise

    async def send(self, payload: dict):
        if not self._started:
            raise RuntimeError(f"[{self.agent_id}] Adaptor not started")

        await self.connection.send(payload)

    async def stop(self):
        if self._started:
            await self.connection.disconnect()
            self._started = False
            logging.info(f"[{self.agent_id}] Adaptor stopped")
