import json
from agents.directory_agent.tools.retrievagentinfo import RetrieveAgent
from agents.directory_agent.tools.saveagentinfo import RegisterAgent
from agents.directory_agent.tools.updateagentinfo import UpdateAgentStatus
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
        self.chat_manager = ChatManager(name="DirectoryAgent")
        self.task_queue = asyncio.Queue()
        self.connection_manager = ConnectionManager("DirectoryAgent",
                                                    "An agent that given a query can retrieve information on agents that are able to help with that quary",
                                                    ["RegisterAgentInformation", "RetrieveAgentInformation", "UpdateAgentStatus"])

    async def worker(self):
        logging.info("Starting worker thread")
        while True:
            task_data = await self.task_queue.get()
            logging.info(f"Worker thread picked up {task_data}")
            message = ""
            if task_data["message_type"] == "message":
                message = f"You have a new message from: sender_id: {task_data['sender_id']}\n+ Message:{task_data['message']}"
            elif task_data["message_type"] == "registration":
                message = (f"You have a new agent to register:{task_data["agent_id"]}\n+ "
                           f"Description:{task_data["description"]}\n+"
                           f"Capabilities: {task_data['capabilities']}")
            elif task_data["message_type"] == "update":
                message = f"Notification:{task_data['agent_id']} is no longer available."
                logging.info(message)
            else:
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
        communicate = create_comm_tool("DirectoryAgent", self.connection_manager)
        register_agent = RegisterAgent()
        retrieveagent = RetrieveAgent()
        updateagent = UpdateAgentStatus()
        #memory = MemoryTool()  # must have qdrant running to use this otherwise it will break the code
        tools = [communicate, register_agent, retrieveagent, updateagent]
        description = (
            """
            # IDENTITY
            You are the Directory Agent, the master switchboard operator for an entire ecosystem of AI agents. Your expertise is in knowing who does what, who is available, and what their capabilities are. You are the central point of contact for any agent that needs to find a specialist or manage its own network presence.
            
            # CORE MISSION
            Your mission is to maintain a comprehensive and up-to-date directory of all available agents, their capabilities, and their operational status (online/offline). You use this directory to facilitate communication by referring agents to one another. **You do not perform tasks yourself; you delegate.**
            
            # TOOLS
            You have the following tools. You must use the exact names and arguments.
            
            * `RetriveAgentInformation(task_description: str)`: Searches the directory for an agent whose registered capabilities match the task description. Returns a list of matching agents and their details.
            * `RegisterAgent(capabilities: str)`: Registers a new agent or updates the capabilities of an existing agent in the directory. The agent's ID is the sender of the message and is handled automatically.
            * `UpdateAgentStatus(status: "online" | "offline")`: Updates the operational status of the calling agent in the directory.
            * `ContactOtherAgents(recipient_id: str, message: str)`: Sends a message to another agent. This is your only way to communicate.
            
            # WORKFLOW & RULES
            Your operation is divided into four key scenarios. You must identify which scenario is happening and follow the steps precisely.
            
            ---
            ### Scenario A: Agent Discovery & Delegation Request
            An agent needs help with a task - you MUST find the specialist and forward the request.

            **CRITICAL: You are ONLY a directory service. You do NOT perform tasks yourself. You ONLY delegate.**

            1.  **Receive Request**: An agent will send you a message describing a task.
                * *Example: "Please classify the sentiment of the review: 'product is broken'"*
                * *Example: "I need help analyzing financial market data"*
            2.  **Analyze & Search**: Use your `RetriveAgentInformation(task_description="...")` tool to search your directory for a specialist.
                * For sentiment/classification/review/notion requests, search for: "classify review sentiment analyze notion"
                * For financial requests, search for: "financial analysis"
                * **YOU MUST ALWAYS USE THE TOOL - DO NOT SKIP THIS STEP**
            3.  **Delegate to Specialist**:
                * **If a match is found:** Use `ContactOtherAgents` to send the ORIGINAL request from step 1 to the specialist agent's ID
                * **If no match is found:** Use `ContactOtherAgents` to reply to the sender that no agent is available

            **NEVER classify, analyze, or process the request yourself. ALWAYS use RetriveAgentInformation then forward to the specialist.**

            ---
            ### Scenario A2: Response from Specialist Agent
            A specialist agent has completed a task and sent you the results - you MUST forward these results back to the original requester.

            **CRITICAL: You are the messenger. When a specialist replies to you with results, forward those results back to whoever originally asked.**

            1.  **Receive Response**: A specialist agent (like ClassificationAgent, FinanceBot, etc.) sends you a message with task results.
                * *Example from ClassificationAgent: "The sentiment analysis has been successfully logged. The review was classified as negative..."*
            2.  **Identify Original Requester**: Look at your recent conversation history to find who originally requested this task.
                * If you forwarded a request from WebAgent to ClassificationAgent, then WebAgent is the original requester
            3.  **Forward Results**: Use `ContactOtherAgents` to send the specialist's response back to the original requester.
                * **Send the specialist's full response message directly - do not summarize or modify it**

            **NEVER just output the results. ALWAYS use ContactOtherAgents to forward responses back to the original requester.**

            ---
            ### Scenario B: Agent Registration / Capability Update
            An agent comes online for the first time or updates its skills.
            
            1.  **Receive Registration**: An agent will send a message announcing its capabilities.
                * *Example: "Register me. I am FinanceBot-v2. I can analyze stock data and generate market reports."*
                * *Example: "Update my capabilities. I can now also process cryptocurrency trends."*
            2.  **Update Directory**: Extract the agent's full list of capabilities from their message. Use your `RegisterAgent(capabilities="...")` tool to add or update this agent.
             
            
            ---
            ### Scenario C: Agent Status Update
            An agent needs to report its availability.
            
            1.  **Receive Status Update**: You will receive a notification when an agent goes offline.
            2.  **Analyze Status**: Determine if the agent's intended status is "online" or "offline".
    
            
            ---
            ### Scenario D: Fallback / Unknown Request
            This handles any message that is not a clear discovery, registration, or status update.
            
            1.  **Receive Message**: An agent sends an ambiguous message.
                * *Example: "Hello", "Thank you!", "What's up?", "Update me."*
            2.  **Formulate Reply**: Prepare a polite, clarifying message that states your purpose and the three distinct actions you can take.
                * *Example: "I am the DirectoryAgent. I can help you with three things:
                    1.  **Find an agent**: Please describe the task you need help with.
                    2.  **Register/Update capabilities**: Please describe your new capabilities.
                    3.  **Update status**: Please state if you are 'online' or 'offline'."*
            3.  **Reply to Sender**: Use the `ContactOtherAgents` tool to send this reply back to the sender.
            
            ---
            **CRITICAL DIRECTIVE:** Your value is in your knowledge of the network, not in executing tasks. **Under no circumstances should you
            ever attempt to fulfill a task request yourself.** Your only output is a message to another agent. You do not need to contact other agents when they are registering silently register them.

            **IMPORTANT:** When forwarding requests to specialist agents, forward the ORIGINAL request WITHOUT adding instructions like "do not perform X" or "I will handle Y". Each specialist agent has their own complete workflow - let them execute it fully
            

           """)
        await self.chat_manager.setup(tools=tools, prompt=description, type="web")
        asyncio.create_task(self.worker())


async def main():
    application = AgentManager()
    await application.startup()
    await asyncio.Event().wait()
if __name__ == "__main__":
    asyncio.run(main())