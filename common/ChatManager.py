from langchain_core.messages import HumanMessage, AIMessage
from common.agent.Agent import Agent
from agents.classification_agent.src.agent.agent_graph import ReviewAgent
from agents.main_agent.backend.agent.Agent import Agent as MainAgent
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
    async def setup(self, tools: List[Any], prompt: str, type: str):
        logging.info("ChatManager setup")
        if type == "web":
            self.type = type
            self.agent = Agent(tools=tools, name=self.name, prompt=prompt)
            self.connection = await aiosqlite.connect(f'db/{self.name}.db')
            self.graph = await self.agent.graph_builder(self.connection)
        elif type == "classify":
            self.type = type
            self.agent = ReviewAgent(name=self.name, tools=tools)
            self.connection = await aiosqlite.connect(f'db/{self.name}.db')
            self.graph = await self.agent.build_graph(self.connection)
        elif type == "main":
            self.type = type
            self.agent = MainAgent(tools=tools, name=self.name, prompt=prompt)
            #self.connection = await aiosqlite.connect(f'db/{self.name}.db')
            state = StateManager.get_state()
            self.graph = await self.agent.graph_builder(state)



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
        if self.type != "main":
            try:
                input_to = {"messages": [HumanMessage(content=user_input)]}
                final_state = await self.graph.ainvoke(input = input_to,
                                                       config={"configurable": {"thread_id": "3"}})
                msg = final_state["messages"][-1].content
                self.messages.append({"role": "agent", "content": msg})
                logging.info(f"Agent output {msg}")
            except Exception as e:
                logging.error(e)
            finally:
                logging.info("Agent finished")
        else:
            try:
                state = StateManager.get_state()
                state.messages.append(HumanMessage(content=user_input))
                updated_state = StateManager.update_state(state)
                finale_state = await self.graph.ainvoke(updated_state, config={"thread_id": "123"})
                if isinstance(finale_state, dict):
                    finale_state= GraphState(**finale_state)

                StateManager.update_state(finale_state)
                msg = finale_state.logs.logs[-1].content
                self.messages.append({"role": "agent", "content": msg})
                logging.info(f"Agent output {msg}")
            except Exception as e:
                logging.error(e)
            finally:
                logging.info("Agent finished")






