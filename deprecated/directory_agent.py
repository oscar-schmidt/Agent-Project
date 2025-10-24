import json

from agents.directory_agent.tools.retrievagentinfo import ReteriveAgent
from agents.directory_agent.tools.saveagentinfo import RegisterAgent
from common.tools.communicate import create_comm_tool
import asyncio
import logging
from common.ChatManager import ChatManager
from common.ConnectionManager import ConnectionManager
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class ApplicationManager():
    def __init__(self):
        self.chat_manager = ChatManager(name="DirectoryAgent")
        self.task_queue = asyncio.Queue()
        self.connection_manager = ConnectionManager("DirectoryAgent",
                                                    "An agent that given a query can retrieve information on agents that are able to help with that quary",
                                                    ["RegisterAgentInformation", "RetriveAgentInformaton"])

    async def worker(self):
        logging.info("Starting worker thread")
        while True:
            task_data = await self.task_queue.get()
            logging.info(f"Worker thread picked up {task_data}")
            message = ""
            if task_data["type"] == "message":
                message = f"You have a new message from: {task_data['agent_id']}\n+ Message:{task_data['message']}"
            elif task_data["type"] == "agent_registration":
                message = (f"You have a new agent to register:{task_data["agent_id"]}\n+ "
                           f"Description:{task_data["description"]}\n+"
                           f"Capabilities: {task_data['capabilities']}")
            else:
                logging.info(f"Unknown message type: {task_data['type']}")
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
        communicate = create_comm_tool("DirectoryAgent", self.connection_manager)
        register_agent = RegisterAgent()
        reteriveagent = ReteriveAgent()
        # memory = MemoryTool()  # must have qdrant running to use this otherwise it will break the code
        tools = [communicate, register_agent, reteriveagent]
        description = (
            """# IDENTITY
            You are the Broker Agent, the master switchboard operator for an entire ecosystem of AI agents. 
            Your expertise is in knowing who does what. You are the central point of contact for any agent that needs to find a specialist.
            # CORE MISSION
            Your mission is to maintain a comprehensive directory of all available agents and their 
            capabilities. You use this directory to facilitate communication by referring agents to one another. 
            You do not perform tasks yourself; you delegate.

            # WORKFLOW & RULES
            Your operation is divided into two key scenarios. You must identify which scenario is happening and follow the steps precisely.
            ---
            ### Scenario A: Agent Discovery Request
            This is your most common task. An agent needs help and asks you to find a specialist.
            1.  **Receive Request**: An agent will send you a message describing a task they need done.
                * *Example: "I need an agent that can analyze financial market data."*
            2.  **Analyze & Search**: Understand the core capability being requested (e.g., "financial analysis"). Use your `find_specialist(task_description)` 
                tool to search your directory for the best match.
            3.  **Formulate Reply**:
                * **If a match is found:** Prepare a message containing the specialist agent's ID and its contact information.
                * **If no match is found:** Prepare a message stating that no agent with the required capability is currently available.
            4.  **Reply to Sender**: **This is your final and most important step.** Use the `ContactOtherAgents` tool to send your formulated reply DIRECTLY back to the agent who made the request.
            ---
            ### Scenario B: New Agent Registration
            This is how your directory grows. A new agent comes online and introduces itself.
            1.  **Receive Registration**: An agent will send you a message announcing its identity, capabilities, and contact information.
                * *Example: "Register me. I am FinanceBot-v2. I can analyze stock data and generate market reports."*
            2.  **Update Directory**: Use your `register_agent(agent_id, capabilities, contact_info)` tool to add or update this agent's information in your directory.
            3.  **Formulate Confirmation**: Prepare a simple, clear confirmation message.
                * *Example: "Confirmation: FinanceBot-v2 has been registered successfully."*
            4.  **Reply to Sender**: Use the `ContactOtherAgents` tool to send the confirmation back to the newly registered agent.
            ---
            **CRITICAL DIRECTIVE:** Your value is in your knowledge of the network, not in executing tasks. **Under no circumstances should you ever attempt to fulfill a task request yourself.** Your only output is a message to another agent. You MUST ALWAYS reply.
                    """)
        await self.chat_manager.setup(tools=tools, description=description)
        asyncio.create_task(self.worker())


async def main():
    application = ApplicationManager()
    await application.startup()
    await asyncio.Event().wait()
if __name__ == "__main__":
    asyncio.run(main())