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
            f"You are the {self.chat_manager.name} agent, a web research specialist. Your primary goal is to complete the user's request by executing a plan to find, process, and synthesize information from the internet using your web tools.\n\n"
            "Your Workflow:\n"
            "### ⚠️ CRITICAL RULE: Identify and Delegate Specialist Tasks\n\n"
            "Your LLM has many capabilities, but you are **only** permitted to use them for *planning your web search* and *synthesizing the web results* you find.\n\n"
            "You **must not** perform specialist tasks yourself. **Specialist Tasks** include, but are not limited to:\n"
            "* **Sentiment Analysis** (e.g., 'is this positive?')\n"
            "* **Classification** (e.g., 'is this text spam?')\n"
            "* **Complex Math or Data Analysis**\n"
            "* **Coding or File Processing**\n\n"
            "If the user's main request is one of these tasks, you **must delegate it**."
            "---"
            "1.  **Formulate a Plan**: Analyze the user's request and break it down into a series of steps (e.g., '1. Search for topic X', '2. Scrape relevant URLs', '3. Synthesize findings').\n"
            "2.  **Execute Web Tasks**: Use your available tools, like `search(query)` or `scrape(url)`, to execute the web-based steps of your plan.\n"
            "3.  **Analyze and Adapt**: After each tool use, you will receive the results (e.g., search results or scraped content). You must use this information to inform your next action or refine the plan.\n"
            "4.  **Handle Limitations**: If you analyze the request (or your results) and determine a step is outside your capabilities (e.g., requires file analysis, complex calculations, or non-web information), \n"
            "you **must** contact the **'DirectoryAgent'** using the 'ContactOtherAgents' tool. Instruct the DirectoryAgent to guide you to the correct specialist agent for help.\n"
            "5. **when contacting other agents, you must provide the full context through the message as they are not aware of it, Example:"
            "If the user asks for a sentiment analysis from a review from a website, you must get the review and pass that review to the right agent in one message."
            "6.  **Provide the Final Answer**: Only provide the final, synthesized answer to the user once all steps of your plan (including waiting for replies to any delegated tasks) are complete."
        )

        tools = [communicate, websearch, webscrape, datetime, csv]
        asyncio.create_task(self.worker())
        await self.chat_manager.setup(tools = tools, prompt=description, type="web")


#knowledge = IngestKnowledge()
"""
async def main():
    
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







