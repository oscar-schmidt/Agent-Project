from typing import List
import websockets
import asyncio
import json
import logging
from websockets import ConnectionClosedError

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class ConnectionManager:
    def __init__(self, agent_id: str, description: str, capabilities: List[str]):
        self.connection = None
        self.uri = "ws://localhost:8765"
        self._websocket: websockets.ClientProtocol | None = None
        self.agent_id = agent_id
        self.description = description
        self.capabilities = capabilities
        self.task = None

    async def connect(self):
        try:
            self._websocket = await websockets.connect(self.uri)
            await self._register()
        except websockets.exceptions.ConnectionClosedError:
            logging.error("Connection Closed")
            self._websocket = None
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Error: {e}")
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
        logging.info(
            f"[{self.agent_id}] Registration response: {registration_response}")
        return registration_response

    async def send_message(self, payload: dict):
        if not self._websocket:
            logging.error("Cannot send message, websocket is closed")
            return

        try:
            await self._websocket.send(json.dumps(payload))
            logging.info(f"[{self.agent_id}] Sent: {payload}")
        except websockets.ConnectionClosedError as e:
            logging.error(f"WebSocket closed while sending: {e}")
            self._websocket = None
            await self.connect()
        except Exception as e:
            logging.error(f"Unexpected error during send: {e}")

    async def start_listening(self, message_handler):
        logging.info("[Adaptor] Listening loop started")

        if not self._websocket:
            raise ConnectionError("Websocket not connected")
        self.task = asyncio.create_task(self.receive(message_handler))

    async def receive(self, message_handler):
        while True:
            if self._websocket is None or self._websocket.closed:
                logging.warning(
                    f"[{self.agent_id}] Websocket closed, reconnecting...")
                await self.connect()
                await asyncio.sleep(5)
                continue

            try:
                message = await self._websocket.recv()
                logging.info(f"[{self.agent_id}] Raw message: {message}")

                task_data = json.loads(message)
                logging.info(f"[{self.agent_id}] Parsed message: {task_data}")

                await message_handler(task_data)
                logging.info(f"[{self.agent_id}] Handler completed")

            except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as error:
                logging.error(f"[{self.agent_id}] Connection error: {error}")
                self._websocket = None
                await asyncio.sleep(5)
            except json.JSONDecodeError as error:
                logging.error(f"[{self.agent_id}] JSON decode error: {error}")
            except Exception as error:
                logging.error(
                    f"[{self.agent_id}] Unexpected error: {error}", exc_info=True)
                self._websocket = None
                await asyncio.sleep(5)

    async def disconnect(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            self._websocket = None
            logging.info(f"[{self.agent_id}] Disconnected")
