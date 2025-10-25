import asyncio
from common.ChatManager import ChatManager
import logging
from common.ConnectionManager import ConnectionManager
from common.tools.communicate import create_comm_tool
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
        communicate = create_comm_tool(
            id="StrategistAgent",
            connection=self.connection_manager,
        )

        tools = [communicate]
        asyncio.create_task(self.worker())
        #asyncio.create_task(self.messanger())
        await self.chat_manager.setup(tools=tools, prompt="", type="main")
