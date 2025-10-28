import asyncio
import websockets
import json
import logging

from websockets import ConnectionClosedOK
from websockets.client import ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedError


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentServer:
    def __init__(self):
        self.agent_registry = {}
        self.queue = asyncio.Queue(maxsize=1000)
        logging.info("Agent server started.")

    async def unregister_agent(self, agent_id: str):
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            await self.update_directory_agent(agent_id, None)
            logging.info(f"Agent {agent_id} was unregistered.")

    async def register_agent(self, agent_id: str, websocket: ClientConnection):
        self.agent_registry[agent_id] = websocket
        logging.info(f"Agent {agent_id} was registered.")
        await websocket.send(json.dumps({"status": "registered"}))

    async def forward_message(self, message_type: str, sender_id: str, recipient_id: str, message: str):
        recipient_ws = self.agent_registry.get(recipient_id)
        if recipient_ws:
            try:
                await recipient_ws.send(json.dumps({
                    "message_type": message_type,
                    "sender_id": sender_id,
                    "message": message,
                }))
                logging.info(
                    f"Agent {message_type}{sender_id} sent a message to {recipient_id}.")
            except ConnectionClosedError:
                logging.warning(f"Agent {sender_id} sending {message} failed")
        else:
            logging.warning(f"Agent {sender_id} sending {message} failed")

    async def dispatch_message(self):
        while True:
            try:
                message_type, recipient_id, sender_id, message = await self.queue.get()
                await self.forward_message(message_type, sender_id, recipient_id, message)
                self.queue.task_done()
            except Exception as e:
                logging.error(f"Error sending message: {e}")

    async def update_directory_agent(self, message, websocket: websockets.ClientProtocol | None):
        data = message
        if websocket is not None:
            try:
                agent_id = data["agent_id"]
            except KeyError:
                logging.error(f"Agent {data["agent_id"]} is missing agent_id.")
                return
            self.agent_registry[agent_id] = websocket
            if agent_id == "DirectoryAgent":
                logging.info(f"Agent {agent_id} was registered.")
                return
            directory_ws = self.agent_registry.get("DirectoryAgent")

            if directory_ws:
                logging.info(
                    f"Notifying Directory Agent about new agent: {agent_id}")
                notification = {
                    "message_type": "registration",
                    "agent_id": agent_id,
                    "description": data.get("description", "No description provided."),
                    "capabilities": data.get("capabilities", [])
                }
                try:
                    logging.info(f"Agent {agent_id} was registered.")
                    await directory_ws.send(json.dumps(notification))
                    # logging.info(f"Agent {agent_id} was registered.")
                except ConnectionClosedError:
                    logging.info(
                        f"Agent {agent_id} sending {notification} failed")
                return
            else:
                logging.info(f"Agent {agent_id} was registered.")
        elif websocket is None:
            try:
                agent_id = data
                directory_ws = self.agent_registry.get("DirectoryAgent")
                if directory_ws:
                    logging.info(
                        f"Notifying Directory Agent that {agent_id} disconnected")
                    notification = {
                        "message_type": "update",
                        "agent_id": agent_id,
                    }
                    try:
                        await directory_ws.send(json.dumps(notification))
                    except ConnectionClosedError:
                        logging.info(
                            f"Agent {agent_id} sending {notification} failed")
                    finally:
                        logging.info(f"Agent {agent_id} was disconnected.")
                        return
            except Exception as e:
                logging.info(f"Error sending message: {e}")
                return
        else:
            logging.info("Wrong message type sent to websocket")
            return

    async def connection_handler(self, websocket):
        agent_id = None
        try:
            registration_message = await websocket.recv()
            data = json.loads(registration_message)

            msg_type = data.get("message_type") or data.get("type")

            if msg_type == "register" and "agent_id" in data:
                agent_id = data["agent_id"]
                logging.info(f"Client Connected and registering: {agent_id}")
                await self.update_directory_agent(message=data, websocket=websocket)
                await websocket.send(json.dumps({
                    "status": "registration successful",
                    "type": "registered"
                }))
            else:
                await websocket.close(1008, json.dumps({"status": "registration failed"}))
                return

            async for message in websocket:
                try:
                    data = json.loads(message)

                    message_type = data.get("message_type") or data.get("type")

                    if message_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                        logging.debug(f"Responded to ping from {agent_id}")
                        continue

                    recipient_id = data.get("recipient_id")
                    sender_id = data.get("sender_id")
                    message_content = data.get(
                        "message") or data.get("content")

                    if recipient_id and message_content:
                        item = (message_type, recipient_id,
                                sender_id, message_content)
                        await self.queue.put(item)
                        logging.debug(
                            f"Queued message from {sender_id} to {recipient_id}")
                    else:
                        logging.warning(
                            f"Invalid message format from {agent_id}: missing required fields. "
                            f"Got: type={message_type}, recipient={recipient_id}, "
                            f"sender={sender_id}, has_content={bool(message_content)}"
                        )

                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Missing required fields: recipient_id or message content"
                        }))

                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error from {agent_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logging.error(
                        f"Error processing message from {agent_id}: {e}")

        except ConnectionClosedError as e:
            logging.info(
                f"Connection closed for agent {agent_id} (error: {e.code})")
        except ConnectionClosedOK:
            logging.info(
                f"Connection closed for agent {agent_id} (normal closure)")
        except Exception as e:
            logging.error(
                f"Connection handler error for agent {agent_id}: {e}")
        finally:
            if agent_id is not None:
                await self.unregister_agent(agent_id)

    async def startup(self):
        logging.info("Server Started startup")
        asyncio.create_task(self.dispatch_message())


async def main():
    server = AgentServer()
    await server.startup()
    async with websockets.serve(server.connection_handler, "localhost", 8765):
        logging.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
