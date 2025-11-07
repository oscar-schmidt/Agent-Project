from agents.classification_agent.src.agent.tools import (
    classify_review_criticality,
    analyze_review_sentiment,
    log_reviews_to_notion,
    get_current_datetime,
    ingest_review
)

from nicegui.events import UploadEventArguments
from common.ChatManager import ChatManager
from nicegui import app, ui
import asyncio
import os
import logging
from common.ConnectionManager import ConnectionManager
from common.tools.communicate import create_comm_tool
#from common.tools.knowledgebase import retriever_tool
#from common.utils import IngestKnowledge
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')




class AgentManager():
    """Manages the agent, however needs to be replaced by the AgentManager
           class from the common directory to reduce duplicate code"""

    def __init__(self):
        self.chat_manager = ChatManager(name="ClassificationAgent")
        self.task_queue = asyncio.Queue()
        self.connection_manager = ConnectionManager("ClassificationAgent",
                                                    "An agent that performs complete review analysis workflow: ingests reviews, classifies criticality, analyzes sentiment, and logs all results to Notion database. This agent handles the ENTIRE workflow end-to-end.",
                                                    ["classify_review_criticality",
                                                     "analyze_review_sentiment",
                                                     "log_reviews_to_notion",
                                                     "get_current_datetime",
                                                     "ingest_review"]
                                                    )
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

#Uncoment the block of code below and comment the main code and if __name__="__main__" code above for individual agent interaction make sure its commented out when running the whole system

"""
application = AgentManager()

@ui.page("/")
def main():
    async def handle_submit():
        text_to_send = user_input.value
        if not text_to_send:
            return
        user_input.value = ''
        application.chat_manager.messages.append({'role': 'user', 'content': text_to_send})
        update_chat_display()
        await application.task_queue.put(text_to_send)
        user_input.value = ''
        update_chat_display()
    async def handle_file_upload(e: UploadEventArguments):
        file_name = e.file.name
        file_path = os.path.join("./knowledge", file_name) #don't upload a file when testing 
        try:
            with open(file_path, 'wb') as file:
                file.write(await e.file.read())
                ui.notify(f"File {file_path} was successfully uploaded")
                await knowledge.ingest_pdf(file_path)
        except Exception as e:
            logging.error(f"Failed to upload file: {e}")


    with ui.column().classes('w-full items-center'):
        ui.label('Welcome').classes('text-2xl mt-4')
    chat_container = ui.column().classes('w-full max-w-2xl mx-auto gap-4 p-4')
    logging.info("Setting up UI")


    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            user_input = ui.input(placeholder="Ask Agent...") \
                .classes('flex-grow').on('keydown.enter', handle_submit)
            submit_button = ui.button('>', on_click=handle_submit)
            submit_button.bind_enabled_from(user_input, 'value')
            ui.upload(
                label="Upload File",
                on_upload=handle_file_upload,
                multiple=False,
                auto_upload=False,
            )
    logging.info("UI setup")

    def update_chat_display():
        chat_container.clear()
        with chat_container:
            for msg in application.chat_manager.messages:
                if msg["role"] == "user":
                    with ui.row().classes('w-full justify-start'):
                        ui.chat_message(msg["content"], name="You", sent=True).classes(
                            'bg-[#2f2f2f] text-gray-200 border border-[#E0E0E0] '
                            'rounded-[20px] px-[15px] py-[10px] m-[5px] max-w-[70%]'
                        )
                elif msg["role"] == "agent":
                    with ui.row().classes('w-full justify-end'):
                        ui.chat_message(msg["content"], name="Agent", sent=False).classes(
                            'bg-[#2d2d2d] text-gray-200 '
                            'rounded-[20px] px-[15px] py-[10px] m-[5px] max-w-[70%]'
                        )

    application.update_ui_callback = update_chat_display
app.on_startup(application.startup)
ui.run()
"""

