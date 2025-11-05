from langmem import create_memory_manager
from .AgentState import State
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode
from typing_extensions import List, Any
from common.memory.semantic import Semantic
from common.memory.episodic import Episode
from config.config_helper import get_model_config
from common.stores.QdrantStore import QdrantStore
import logging

logging.basicConfig(level=logging.INFO)

class Agent:
    """An Agent class"""
    def __init__(self, tools: List[Any],name: str, prompt: str):
        """init method for class Agent requires tools as a list of tools"""
        model_config = get_model_config()
        self.name = name
        self.prompt= prompt
        self.llm = init_chat_model(f"{model_config["provider"]}:{model_config['model']}")
        self.tools = tools
        self.llm_with_tools  = self.llm.bind_tools(self.tools)
        self.manager =  create_memory_manager(
                "gpt-4o-mini",
                schemas=[Episode, Semantic],
                instructions="Extract all user information and events as Episodes, and any facts as Semantic",
            )
        #self.store = QdrantStore(collection_name="WebAgent") #don't uncomment if you don't have qdrant running

    def planner(self, state: State):
        if self.name == "WebAgent":
            planner_messages =  [("user", f"{state["messages"][-1].content}")] + [
                (
                    "system",
                        f"You are an expert planner for a review for {self.name}. "
                        "Create a concise, step-by-step plan to answer the user/agents request. "
                        "Consider which tools to use and in what order. "
                        "If a message has come from another agent you must make sure that the contactotheragents tool is used to update the other agent"
                        f"Available tools: {[tool.name for tool in self.tools]}\n"
                )
            ]
        else:
            planner_messages = [("user", f"{state["messages"][-1].content}")] + [
                (
                    "system",
                    f"You are an expert planner for a review for {self.name}. "
                    "Create a concise, step-by-step plan to answer the user/agents request. "
                    "Consider which tools to use and in what order. "
                    "Important do not use contactotheragents tool whenever you are registering an agent simply register the agent silently"
                    f"Available tools: {[tool.name for tool in self.tools]}\n"
                )
            ]
        plan = self.llm_with_tools.invoke(planner_messages).content

        state["plan"] = plan

        return state

    def chat(self, state: State):
        system_prompt = (
           self.prompt + f"You must follow this plan {state['plan']}"
        )

        if state.get('critique') and state['critique'] != 'None':
            system_prompt += f"you must revise your previous answer based on the following critique: {state['critique']}"
        messages_with_prompt = [("system", system_prompt)] + state["messages"]
        ai_response = self.llm_with_tools.invoke(messages_with_prompt)
        return {"messages":[ai_response]}


    def critique(self, state: State):
        critique_prompt = (
            "You are an expert critic, review the proposed final answer answer to the original user request."
            "Is the answer complete, accurate, and does it fully address the user's query?"
            "If the answer is good respond with 'yes'."
            "If it needs improvement, provide a concise critique of whats missing or wrong."
            f"This is the plan {state["plan"]}"
        )
        critique_message = [
            state["messages"][-1],
            ("system", critique_prompt)
        ]
        #not critiquing at the moment because its brittle and can get stuck in a recursive loop
        critique = "yes" #self.llm_openai_tools.invoke(critique_message).content
        print(f' \nPrinting critique: \n {critique}')
        if "yes" in critique.lower():
            return {'critique':'None'}
        else:
            return {'critique': critique}

    def route_tools(self, state: State):
        ai_message = state["messages"][-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "critique"


    def should_continue(self, state: State):
        if state.get("critique") and state["critique"] != "None":
            return "chatbot"
        else:
            # extracted_memory = self.manager.invoke({"messages": state["messages"]})
            # self.store.put(extracted_memory)
            return END


    async def graph_builder(self, connection):
        graph_builder = StateGraph(State)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("chatbot", self.chat)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("critique", self.critique)

        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("planner", "chatbot")
        graph_builder.add_edge("tools", "chatbot")

        graph_builder.add_conditional_edges(
            "chatbot",
            self.route_tools,
            {"tools": "tools", "critique": "critique"},
        )


        graph_builder.add_conditional_edges(
            "critique",
            self.should_continue,
            {"chatbot":"chatbot", "__end__": END}
        )

        memory = AsyncSqliteSaver(connection)
        return graph_builder.compile(checkpointer=memory)
