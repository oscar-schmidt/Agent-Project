#from backend.model.states.StateManager import StateManager
from agents.main_agent.frontend.home_ui import render_chat_section, render_file_uploader, render_sidebar
from typing import Any
from nicegui import ui, app
from langgraph.checkpoint.serde import jsonplus
from langgraph.checkpoint.serde.jsonplus import _msgpack_default
from langgraph.checkpoint.serde.jsonplus import _option
from langgraph.checkpoint.serde.jsonplus import ormsgpack
import asyncio
from common.ChatManager import ChatManager
import logging
from common.ConnectionManager import ConnectionManager
from common.tools.knowledgebase import retriever_tool

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentManager():
    """Manages the agent, however needs to be replaced by the AgentManager
        class from the common directory to reduce duplicate code"""
    def __init__(self):
        self.connection_manager = ConnectionManager(
            agent_id="StrategistAgent",
            description="An agent that can analyze, plan, and strategize using advanced reasoning skills",
            capabilities=["DatabaseRetrival", "KnowledgeRetrival"])
        self.chat_manager = ChatManager(name="StrategistAgent")
        self.task_queue = asyncio.Queue()
        self.user_input = None
        self.update_ui_callback = None

    async def worker(self):
        logging.info("Starting worker thread")
        while True:
            task_data = await self.task_queue.get()
            logging.info(f"Worker thread picked up {task_data}")
            logging.info(type(self.chat_manager))
            message = ""
            if isinstance(task_data, dict):
                message = f"You have a new message from: {task_data['sender_id']}\n+ Message:{task_data['message']}"
            elif isinstance(task_data, str):
                message = task_data
            else:
                logging.info(f"Incorrect message format")
            await self.chat_manager.run_agent(message)
            self.task_queue.task_done()
            if self.update_ui_callback:
                self.update_ui_callback()

    async def message_handler(self, message: dict):
        await self.task_queue.put(message)



    async def startup(self):
        try:
            await self.connection_manager.connect()
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return
        await self.connection_manager.start_listening(message_handler=self.message_handler)




        tools = [retriever_tool]
        asyncio.create_task(self.worker())
        #asyncio.create_task(self.messanger())
        await self.chat_manager.setup(tools=tools, prompt="", type="main")



# Reference from: https://github.com/langchain-ai/langgraph/issues/4956#issuecomment-3135374853
def message_to_dict(msg):
    """
    Recursively convert a message or object into a dict/str (safe for serialization).
    """
    if hasattr(msg, "to_dict"):
        return msg.to_dict()
    elif isinstance(msg, dict):
        return {k: message_to_dict(v) for k, v in msg.items()}
    elif isinstance(msg, (list, tuple)):
        return [message_to_dict(x) for x in msg]
    elif isinstance(msg, (str, int, float, bool, type(None))):
        return msg
    else:
        print("Serialization Fallback, type:", type(msg))
        print(msg)
        return {"role": getattr(msg, "role", "user"), "content": str(getattr(msg, "content", msg))}


def _msgpack_enc(data: Any) -> bytes:
    return ormsgpack.packb(message_to_dict(data), default=_msgpack_default, option=_option)


def monkey_patch():
    setattr(jsonplus, "_msgpack_enc", _msgpack_enc)

def start():
    monkey_patch()

    with ui.row().classes('w-full flex h-screen p-6 gap-6'):

        with ui.column().classes(' flex-3 w-1/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_sidebar()

        with ui.column().classes('flex-9  w-3/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_chat_section()

agent = AgentManager()
async def startup_agent():
    await agent.startup()
asyncio.run(startup_agent())
start()
ui.run(title='Graph AI Chat', reload=True, port=8080)
