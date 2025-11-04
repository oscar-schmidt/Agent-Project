import asyncio
import json
import logging
from agents.main_agent.Server.adaptor import Adaptor

logging.basicConfig(level=logging.INFO)


class MainAgentAdaptor(Adaptor):
    @classmethod
    async def create(cls):
        logging.info("[MainAgentAdaptor] Creating adaptor...")

        instance = await cls.get_adaptor(
            agent_id="main_agent",
            description="Interactive client",
            capabilities=["qa", "summary"]
        )

        if not instance._started:
            await instance.start(instance.handle_message)
            logging.info("[MainAgentAdaptor] Adaptor started and listening")

        return instance

    async def handle_message(self, message: dict):
        try:
            logging.info(
                f"[MainAgentAdaptor] handle_message called with: {message}")
            await self.message_queue.put(message)
            logging.info(
                f"[MainAgentAdaptor] Message queued, queue size: {self.message_queue.qsize()}")
        except Exception as e:
            logging.error(
                f"[MainAgentAdaptor] handle_message failed: {e}", exc_info=True)

    async def send_message(self, recipient_id: str, message: str):
        payload = {
            "message_type": "message",
            "recipient_id": recipient_id,
            "sender_id": self.agent_id,
            "message": message,
        }

        logging.info(
            f"[MainAgentAdaptor] Sending to {recipient_id}: {message}")

        try:
            await self.connection.send(payload)
            logging.info("[MainAgentAdaptor] Message sent successfully")
        except Exception as e:
            logging.error(
                f"[MainAgentAdaptor] Send failed: {e}", exc_info=True)
            raise

    async def receive_message(self, timeout: float = 60.0):
        logging.info(
            f"[MainAgentAdaptor] Waiting for message (timeout={timeout}s)")
        logging.info(
            f"[MainAgentAdaptor] Current queue size: {self.message_queue.qsize()}")

        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout)
            logging.info(
                f"[MainAgentAdaptor] Retrieved message from queue: {message}")
            return json.dumps(message)
        except asyncio.TimeoutError:
            logging.warning("[MainAgentAdaptor] Timeout - no message received")
            return None
        except Exception as e:
            logging.error(
                f"[MainAgentAdaptor] Receive error: {e}", exc_info=True)
            return None

    async def disconnect(self):
        logging.info("[MainAgentAdaptor] Disconnecting...")
        await self.connection.disconnect()
        self._started = False
        logging.info("[MainAgentAdaptor] Disconnected")
