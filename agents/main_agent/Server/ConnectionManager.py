import asyncio
import json
import logging
from typing import List, Optional
import websockets

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


class ConnectionManager:
    def __init__(self, agent_id: str, description: str, capabilities: List[str]):
        self.uri = "ws://localhost:8765"
        self.agent_id = agent_id
        self.description = description
        self.capabilities = capabilities

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_connected = False
        self.reconnect_delay = 2

    async def connect(self):
        while not self.is_connected:
            try:
                self.websocket = await websockets.connect(
                    self.uri, ping_interval=30, ping_timeout=15
                )
                self.is_connected = True
                logging.info(f"Connected to {self.uri}")

                await self._send_register()

                asyncio.create_task(self._receive_loop())
                asyncio.create_task(self._heartbeat())
            except Exception as e:
                logging.error(
                    f"Connection failed: {e}. Retrying in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    async def _send_register(self):
        payload = {
            "message_type": "register",
            "agent_id": self.agent_id,
            "description": self.description,
            "capabilities": self.capabilities,
        }
        await self._send_json(payload)
        logging.info(f"Registering agent: {self.agent_id}")

    async def send(self, message: dict):
        await self._send_json(message)
        logging.info("Message sent")

    async def _send_json(self, data: dict):
        if not self.websocket or not self.is_connected:
            logging.warning("⚠️ Not connected, attempting reconnect...")
            await self._reconnect()
        try:
            await self.websocket.send(json.dumps(data))
        except Exception as e:
            logging.error(f"Send failed: {e}")
            await self._reconnect()

    async def start_listening(self, handler):
        while True:
            msg = await self.message_queue.get()
            try:
                await handler(msg)
            except Exception as e:
                logging.error(f"Error in message handler: {e}")

    async def _receive_loop(self):
        try:
            async for raw in self.websocket:
                try:
                    data = json.loads(raw)
                    msg_type = data.get("message_type") or data.get("type")

                    if msg_type in ["ping", "pong", "register", "registered", "heartbeat"]:
                        continue

                    await self.message_queue.put(data)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON: {raw}")
        except Exception as e:
            logging.warning(f"Connection closed ({e}), reconnecting...")
            await self._reconnect()

    async def _heartbeat(self):
        while self.is_connected and self.websocket:
            await asyncio.sleep(60)
            try:
                await self.websocket.send(json.dumps({"type": "ping"}))
            except Exception:
                await self._reconnect()

    async def _reconnect(self):
        self.is_connected = False
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
        logging.info(f"Reconnecting in {self.reconnect_delay}s...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, 30)
        await self.connect()

    async def disconnect(self):
        self.is_connected = False
        if self.websocket:
            try:
                await self.websocket.close()
                logging.info("Disconnected from server")
            except Exception as e:
                logging.error(f"Error during disconnect: {e}")
            finally:
                self.websocket = None
