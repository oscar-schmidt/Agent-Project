import asyncio
import json
import logging
from common.ConnectionManager import ConnectionManager


logging.basicConfig(level=logging.INFO)


async def handle_message(message_data: dict, connection: ConnectionManager):
    logging.info(f"[handle_message] Called!")
    logging.info(f"[handle_message] message_data type: {type(message_data)}")
    logging.info(f"[handle_message] message_data value: {message_data}")
    logging.info(f"[handle_message] connection type: {type(connection)}")
    msg = message_data.get("message")
    sender_id = message_data.get("sender_id")
    recipient_id = message_data.get("recipient_id")
    msg_type = message_data.get("message_type")

    logging.info(
        f"WebAgent received from {sender_id} / {recipient_id} message_type: {msg_type}: {msg}")

    await connection.send({
        "message_type": "message",
        "recipient_id": "main_agent",
        "sender_id": "WebAgent",
        "message": f"weather for tommorrow is 18 degree."
        # "message": f"WebAgent received your message from  {sender_id} / {agent_id} message_type: {msg_type}: {msg}"
    })


async def test(connection):
    await connection.connect()
    await connection.send({
        "message_type": "message",
        "recipient_id": "main_agent",
        "sender_id": "WebAgent",
        "message": "hello"
    })


async def main():
    connection = ConnectionManager(
        agent_id="WebAgent",
        description="WebAgent",
        capabilities=["online search"]
    )

    try:
        await connection.connect()
        # await test(connection)
        logging.info("WebAgent connected and registered")

        await connection.start_listening(
            lambda message_data: handle_message(message_data, connection)
        )

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"WebAgent encountered an error: {e}")

    finally:
        await connection.disconnect()
        logging.info("WebAgent disconnected")


asyncio.run(main())
