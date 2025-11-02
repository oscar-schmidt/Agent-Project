import asyncio
import json
import logging
from functools import partial

from agents.main_agent.Server.ConnectionManager import ConnectionManager

logging.basicConfig(level=logging.INFO)


async def handle_message(message_data: dict, connection: ConnectionManager):
    msg = message_data.get("message")
    sender_id = message_data.get("sender_id")
    agent_id = message_data.get("agent_id")
    msg_type = message_data.get("message_type")

    logging.info(
        f"DirectoryAgent received from {sender_id} / {agent_id} message_type: {msg_type}: {msg}")

    await connection.send_message({
        "message_type": "message",
        "recipient_id": "main_agent",
        "sender_id": "WebAgent",
        "message": f"weather for tommorrow is 18 degree."
        # "message": f"WebAgent received your message from  {sender_id} / {agent_id} message_type: {msg_type}: {msg}"
    })


async def main():
    connection = ConnectionManager(
        agent_id="WebAgent",
        description="WebAgent",
        capabilities=["online search"]
    )

    try:
        await connection.connect()
        logging.info("WebAgent connected and registered")

        await connection.start_listening(
            partial(handle_message, connection=connection)
        )

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"WebAgent encountered an error: {e}")

    finally:
        await connection.disconnect()
        logging.info("WebAgent disconnected")


asyncio.run(main())
