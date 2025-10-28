from langchain_core.messages import HumanMessage, AIMessage
from common.agent.Agent import Agent
from agents.classification_agent.src.agent.agent_graph import ReviewAgent
from agents.main_agent.backend.graph.get_graph import MainAgent
import aiosqlite
from typing import List, Any
import logging
from typing import Any, List
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.StateManager import StateManager
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class ChatManager:
    def __init__(self, name: str):
        self.messages = []
        self.agent = None
        self.graph = None
        self.connection = None
        self.websocket = None
        self.name = name
        self.type = None

    async def setup(self, tools: List[Any], prompt: str, type: str, websocket: Any | None = None):
        logging.info("ChatManager setup")
        self.connection = await aiosqlite.connect(f'db/{self.name}.db')
        if type == "web":
            self.type = type
            self.agent = Agent(tools=tools, name=self.name, prompt=prompt)
            self.graph = await self.agent.graph_builder(self.connection)
        elif type == "classify":
            self.type = type
            self.agent = ReviewAgent(name=self.name, tools=tools)
            self.graph = await self.agent.build_graph(self.connection)
        elif type == "main":
            self.type = type
            self.agent = MainAgent(tools=tools)
            self.graph = await self.agent.get_graph(connection=self.connection)



    def load_message(self, messages):
        for message in messages["messages"]:
            if isinstance(message, HumanMessage):
                if message.content:
                    self.messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                if message.content:
                    self.messages.append({"role": "agent", "content": message.content})
            else:
                continue

    async def run_agent(self, user_input):
        try:
            logging.info(f"Agent input {user_input}")
            input_to = {"messages": [HumanMessage(content=user_input)]}
            final_state = await self.graph.ainvoke(input_to, config={"thread_id": "123"})
            msg = final_state['messages'][-1].content
            self.messages.append({"role": "agent", "content": msg})
            logging.info(final_state)
        except Exception as e:
            logging.error(e)
        finally:
            logging.info("Agent finished")





