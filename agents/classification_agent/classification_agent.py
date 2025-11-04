from agents.classification_agent.src.agent.tools import (
    classify_review_criticality,
    analyze_review_sentiment,
    log_reviews_to_notion,
    get_current_datetime,
    ingest_review
)
from common.tools.communicate import create_comm_tool
import asyncio
import logging
from common.ChatManager import ChatManager
from common.ConnectionManager import ConnectionManager

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentManager():
    """Manages the agent, however needs to be replaced by the AgentManager
           class from the common directory to reduce duplicate code"""

    def __init__(self):
        self.chat_manager = ChatManager(name="ClassificationAgent")
        self.task_queue = asyncio.Queue()
        self.connection_manager = ConnectionManager("ClassificationAgent",
                                                    "An agent that can classify reviews, analyze sentiment, and ingest reviews from other agents",
                                                    ["classify_review_criticality",
                                                     "analyze_review_sentiment",
                                                     "log_reviews_to_notion",
                                                     "get_current_datetime",
                                                     "ingest_review"]
                                                    )

    async def worker(self):
        logging.info("Starting worker thread")
        while True:
            task_data = await self.task_queue.get()
            logging.info(f"Worker thread picked up {task_data}")
            message = ""
            if task_data["message_type"] == "message":

                message = f"You have a new message from: {task_data['sender_id']}\n+ Message:{task_data['message']}"
                logging.info(message)
            else:
                message = f"You have a new message from: {task_data['sender_id']}\n+ Message:{task_data['message']}"
                logging.info(f"Unknown message type: {task_data['message_type']}")
            await self.chat_manager.run_agent(message)
            self.task_queue.task_done()

    async def message_handler(self, message: dict):
        await self.task_queue.put(message)

    async def startup(self):
        try:
            await self.connection_manager.connect()
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return
        await self.connection_manager.start_listening(message_handler=self.message_handler)
        communicate = create_comm_tool("ClassificationAgent", self.connection_manager)
        # memory = MemoryTool()  # must have qdrant running to use this otherwise it will break the code
        tools = [
                communicate,
                classify_review_criticality,
                analyze_review_sentiment,
                log_reviews_to_notion,
                get_current_datetime,
                ingest_review
        ]

        await self.chat_manager.setup(tools=tools, prompt="", type="classify")
        asyncio.create_task(self.worker())


async def main():
    application = AgentManager()
    await application.startup()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())