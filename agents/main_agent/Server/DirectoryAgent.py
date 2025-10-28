import asyncio
import json
import logging
from functools import partial
from ConnectionManager import ConnectionManager

logging.basicConfig(level=logging.INFO)


async def handle_message(message_data: dict, connection: ConnectionManager):
    msg = message_data.get("message")
    sender_id = message_data.get("sender_id")
    agent_id = message_data.get("agent_id")
    msg_type = message_data.get("message_type")

    logging.info(
        f"DirectoryAgent received from {sender_id} / {agent_id} message_type: {msg_type}: {msg}")

    await connection.send({
        "message_type": "message",
        "recipient_id": "main_agent",
        "sender_id": "DirectoryAgent",
        "message": f"DirectoryAgent received your message from  {sender_id} / {agent_id} message_type: {msg_type}: {msg}"
    })


async def main():
    connection = ConnectionManager(
        agent_id="DirectoryAgent",
        description="Directory and routing agent",
        capabilities=["routing", "directory"]
    )

    try:
        await connection.connect()
        logging.info("DirectoryAgent connected and registered")

        await connection.start_listening(
            partial(handle_message, connection=connection)
        )

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"DirectoryAgent encountered an error: {e}")

    finally:
        await connection.disconnect()
        logging.info("DirectoryAgent disconnected")


if __name__ == "__main__":
    asyncio.run(main())
