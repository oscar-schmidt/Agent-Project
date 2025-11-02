import asyncio
import json
import logging
from agents.main_agent.Server.Adaptor import Adaptor

logging.basicConfig(level=logging.INFO)


class MainAgentAdaptor(Adaptor):
    @classmethod
    async def create(cls):
        logging.info("[MainAgentAdaptor] Starting adaptor...")

        instance = await cls.get_adaptor(
            agent_id="main_agent",
            description="Interactive client",
            capabilities=["qa", "summary"]
        )
        if not instance._started:
            await instance.connection.connect()
            asyncio.create_task(
                instance.connection.start_listening(instance.handle_message))
            logging.info(
                "[MainAgentAdaptor] Connected and listening started.")

            instance._keep_alive_task = asyncio.create_task(
                instance._keep_alive())
            instance._started = True

        return instance

    async def _keep_alive(self):
        while True:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logging.info("[MainAgentAdaptor] Keep-alive cancelled.")
                break

    async def handle_message(self, message: dict):
        logging.info(f"[MainAgentAdaptor] Received: {message}")
        await self.message_queue.put(message)

    async def send_message(self, recipient_id: str, message: str):
        payload = {
            "message_type": "message",
            "recipient_id": recipient_id,
            "sender_id": "main_agent",
            "message": message,
        }

        logging.info(
            f"[MainAgentAdaptor] Sending to {recipient_id}: {message}")
        try:
            await self.connection.send_message(payload)
            logging.info("[MainAgentAdaptor] Message sent successfully.")
        except Exception as e:
            logging.error(
                f"[MainAgentAdaptor] Send failed: {e}", exc_info=True)

    async def receive_message(self, timeout: float = 30.0):
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout)
            logging.info(
                f"[MainAgentAdaptor] Got message from queue: {message}")
            return json.dumps(message)
        except asyncio.TimeoutError:
            logging.warning(
                "[MainAgentAdaptor] Timeout waiting for message.")
            return None
        except Exception as e:
            logging.error(
                f"[MainAgentAdaptor] Receive failed: {e}", exc_info=True)
            return None

    async def disconnect(self):
        logging.info("[MainAgentAdaptor] Disconnecting...")
        await self.connection.disconnect()
        logging.info("[MainAgentAdaptor] Disconnected.")
