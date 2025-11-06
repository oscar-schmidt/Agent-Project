from nicegui.events import UploadEventArguments
from common.ChatManager import ChatManager
import websockets
from nicegui import app, ui
import asyncio
import json
import os
import logging
from common.ConnectionManager import ConnectionManager
from agents.web_agent.tools.webscrape import WebScrape
from agents.web_agent.tools.websearch import WebSearch
from common.tools.memorytool import MemoryTool
from common.tools.databse import DatabaseTool
from common.tools.date_time import DateTime
from common.tools.communicate import create_comm_tool
from common.tools.csv import CSVTool
#from common.tools.knowledgebase import retriever_tool
#from common.utils import IngestKnowledge
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class AgentManager():
    """Manages the agent, however needs to be replaced by the AgentManager
    class from the common directory to reduce duplicate code"""
    def __init__(self):
        self.connection_manager = ConnectionManager(
            agent_id="WebAgent",
            description="A specialized agent for web-based information retrieval. It uses WebSearch to find relevant pages and Webscrape to extract specific data.",
            capabilities=["WebSearch", "Webscrape"])
        self.chat_manager = ChatManager(name="WebAgent")
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

        communicate = create_comm_tool("WebAgent", self.connection_manager)
        websearch = WebSearch()
        webscrape = WebScrape()
        datetime = DateTime()
        #memory = MemoryTool()
 
        # database = DatabaseTool()
        csv = CSVTool()
        description = (
            f"You are the {self.chat_manager.name} agent, a specialized Web Research Specialist. "
            "Your sole purpose is to process information requests from users and other agents by finding, "
            "synthesizing, and—crucially—DELIVERING that information back to the requester.\n\n"

            "**Your Toolkit:**\n"
            "- WebSearch: Google Search for information\n"
            "- Webscrape: Extract content from specific URLs\n"
            "- DateTime: Get current date/time for context\n"
            "- CSV: Read/write CSV data\n"
            "- communicate: Send messages to other agents/users\n\n"

            "**PROTOCOL 1: IMMEDIATE HANDOFF TRIGGERS (Highest Priority)**\n"
            "If the request contains specific keywords unrelated to pure research, you MUST immediately hand off the task.\n"
            "- **Keywords:** 'classify', 'sentiment', 'notion', 'review', 'log', 'inventory', 'sales'\n"
            "- **Action:** Use `communicate` tool → Send EXACT request to 'DirectoryAgent'.\n"
            "- **End State:** Do nothing else after this handoff.\n\n"

            "**PROTOCOL 2: STANDARD RESEARCH LOOP (standard priority)**\n"
            "For all OTHER requests that do not trigger Protocol 1:\n"
            "1. **Identify Sender:** Note the `[sender_id]` of the incoming message.\n"
            "2. **Execute Research:** Use your WebSearch/Webscrape tools as necessary to find the answer. You are authorized to use these tools even if not explicitly requested, if they are needed to answer the question.\n"
            "3. **Synthesize:** Summarize the findings clearly.\n"
            "4. **MANDATORY FINAL STEP - REPLY:** You MUST use the `communicate` tool to send your synthesized findings back to the original `[sender_id]`. Your task is NOT complete until you have sent this reply.\n\n"

            "**CRITICAL CONSTRAINTS:**\n"
            "- NEVER attempt to classify sentiment or reviews yourself.\n"
            "- NEVER contact 'NotionAgent' directly (always use DirectoryAgent for this).\n"
            "- IF you receive a message, YOU MUST PROVIDE A REPLY. Leaving a message unacknowledged is a critical failure."
        )

        tools = [communicate, websearch, webscrape, datetime, csv]
        asyncio.create_task(self.worker())
        await self.chat_manager.setup(tools = tools, prompt=description, type="web")


#knowledge = IngestKnowledge()

async def main():
    application = AgentManager()
    await application.startup()
    await asyncio.Event().wait()
if __name__ == "__main__":
    asyncio.run(main())
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
        file_path = os.path.join("./knowledge", file_name)
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






