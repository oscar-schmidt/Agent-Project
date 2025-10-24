import asyncio
import websockets
import json


async def send_messages(websocket, agent_id):
    while True:
        message = input("Enter message (or 'quit' to exit): ").strip()
        if message.lower() == "quit":
            print("Exiting...")
            await websocket.close()
            break

        recipient_id = input("Enter recipient_id: ").strip()

        data = {
            "message_type": "chat",
            "sender_id": agent_id,
            "recipient_id": recipient_id,
            "message": message,
        }

        await websocket.send(json.dumps(data))
        print(f"Sent message to {recipient_id}")


async def receive_messages(websocket):
    """Continuously receive messages from the server."""
    while True:
        try:
            async for msg in websocket:
                print(f"\nReceived from server: {msg}")
        except websockets.ConnectionClosed:
            print("Server closed the connection.")


async def main():
    uri = "ws://localhost:8765"
    agent_id = input("Enter your agent_id: ").strip()
    connection = await websockets.connect(uri)
    await connection.send(json.dumps({
        "message_type": "register",
        "agent_id": agent_id,
        "description": "Interactive client",
        "capabilities": ["chat"]
    }))

    response = await connection.recv()
    print(f"Registration response: {response}")

    # Run sender and receiver concurrently
    receiver_task = asyncio.create_task(receive_messages(connection))
    sender_task = asyncio.create_task(
        send_messages(connection, agent_id))

    done, pending = await asyncio.wait(
        [receiver_task, sender_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
