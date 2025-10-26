from typing import List
import websockets
import asyncio
import json
import logging
from websockets import ConnectionClosedError

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class ConnectionManager:
    def __init__(self, agent_id: str, description: str, capabilities: List[str],):
        self.connection = None
        self.uri = "ws://localhost:8765"
        self._websocket: websockets.ClientProtocol | None = None
        self.agent_id = agent_id
        self.description = description
        self.capabilities = capabilities

    async def connect(self):
        try:
            self._websocket = await websockets.connect(self.uri)
            await self._register()
        except websockets.exceptions.ConnectionClosedError:
            logging.info("Connection Closed")
            self.websocket = None
            await asyncio.sleep(5)
        except Exception as e:
            logging.info(f"Error: {e}")
            self._websocket = None
            await asyncio.sleep(5)

    async def _register(self) -> str:
        if not self._websocket:
            raise ConnectionError("Websocket not connected")

        await self._websocket.send(json.dumps({
            "message_type": "register",
            "agent_id": self.agent_id,
            "description": self.description,
            "capabilities": self.capabilities,

        }))
        registration_response = await self._websocket.recv()
        logging.info(registration_response)

    async def send(self, payload):
        logging.info(f"Sending message")
        if not self._websocket:
            logging.error("Websocket not connected")
            return "Websocket not connected"

        await self._websocket.send(json.dumps(payload))

    async def start_listening(self, message_handler):
        if not self._websocket:
            raise ConnectionError("Websocket not connected")
        self.task = asyncio.create_task(self.receive(message_handler))

    async def receive(self, message_handler):
        while True:
            if not self._websocket:
                await self.connect()
            try:
                async for message in self._websocket:
                    data = json.loads(message)
                    await message_handler(data)
            except (websockets.ConnectionClosedError, ConnectionResetError):
                logging.warning("WebSocket closed, reconnecting...")
                self._websocket = None
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Receive error: {e}")
                self._websocket = None
                await asyncio.sleep(5)

    async def close(self):
        self._websocket.close()
        self._websocket = None
