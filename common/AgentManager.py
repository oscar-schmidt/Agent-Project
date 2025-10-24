from common.ChatManager import ChatManager
import websockets
from typing import List, Any
import asyncio
import json
import logging
from common.ConnectionManager import ConnectionManager
from agents.web_agent.tools.webscrape import WebScrape
from agents.web_agent.tools.websearch import WebSearch
from common.tools.memorytool import MemoryTool
from common.tools.databse import DatabaseTool
from common.tools.date_time import DateTime
from common.tools.communicate import create_comm_tool
from common.tools.csv import CSVTool

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentManager():
    def __init__(self,  agent_id: str, prompt: str,tools: List[Any], description: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.prompt = prompt
        self.description = description
        self.capabilities = capabilities
        self.tools = tools
        self.connection_manager = ConnectionManager(
            agent_id=self.agent_id,
            description=self.description,
            capabilities=self.capabilities)
        self.chat_manager = ChatManager(name=self.agent_id)
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
                message = f"You have a new message from: {task_data['sender']}\n+ Message:{task_data['message']}"
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

        asyncio.create_task(self.worker())
        await self.chat_manager.setup(tools = self.tools, prompt=self.prompt)