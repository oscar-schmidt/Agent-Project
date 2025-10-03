from langchain_core.messages import HumanMessage
from langmem import create_memory_manager
from .AgentState import State
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from typing_extensions import List, Any
from src.backend.memory.semantic import Semantic
from src.backend.memory.episodic import Episode
from src.backend.memory.QdrantStore import QdrantStore
import sqlite3
class Agent:
    """An Agent class"""
    def __init__(self, tools: List[Any], name: str):
        """init method for class Agent requires tools as a list of tools"""
        self.name = name
        self.llm = init_chat_model("ollama:qwen3:8b") #updated this for ollama as "ollama:model_name"
        self.llm_openai = init_chat_model("gpt-4o-mini")
        self.tools = tools
        self.llm_with_tools  = self.llm.bind_tools(self.tools)
        self.llm_openai_tools = self.llm_openai.bind_tools(self.tools)
        self.manager =  create_memory_manager(
                "gpt-4o-mini",
                schemas=[Episode, Semantic],
                instructions="Extract all user information and events as Episodes, and any facts as Semantic",
            )
        #self.store = QdrantStore(collection_name="WebAgent") #don't uncomment if you don't have qdrant running

    def run(self, user_input: str):
        """Method to run the agent/interact with the agent requires user input"""
        #Shows the final message from the LLM
        with sqlite3.connect(f"{self.name}.db") as conn:
            graph = self.graph_builder()
            memory = SqliteSaver(conn)
            compiled = graph.compile(checkpointer=memory)
            input = {"messages": [HumanMessage(content=user_input)]}
            print("Just before invoke")
            final_state = compiled.invoke(
                input,
                config=self.config
            )
            return final_state["messages"][-1].content


    def planner(self, state: State):
        state["tools"] = self.tools

        planner_messages =  [("user", f"{state["messages"][-1].content}")] + [
            (
                "system",
                "You are an expert planner, create a concise, step-by-step plan to answer the user's request. Respond with the plan only"
                f"These are the tools available for use {state["tools"]}"
            )
        ]
        plan = self.llm_openai_tools.invoke(planner_messages).content
        state["plan"] = plan

        return state

    def chat(self, state: State):
        system_prompt = (
            "You are a helpful assistant. Your goal is to answer the user's request by following the plan.\n"
            "1. First, decide if you need to call a tool to execute the current step of the plan.\n"
            "2. If you do, call the tool. \n"
            "3. **Crucially, when you receive the result from a tool (a ToolMessage), you MUST use that result to form a final answer for the user.**\n"
            "4. Do NOT call the same tool again unless the plan explicitly requires it for a different step. Once you have the information, give the answer.\n"
            f"\nHere is the plan to follow: {state.get('plan', 'No plan available')}"
        )
        if state.get('critique') and state['critique'] != 'None':
            system_prompt += f"you must revise your previous answer based on the following critique: {state['critique']}"
        messages_with_prompt = [("system", system_prompt)] + state["messages"]
        ai_response = self.llm_openai_tools.invoke(messages_with_prompt)
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

        critique = self.llm_openai_tools.invoke(critique_message).content
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
            #extracted_memory = self.manager.invoke({"messages": state["messages"]})
            #self.store.put(extracted_memory)
            return END

    def graph_builder(self):
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

        with sqlite3.connect('src/backend/db/WebAgent.db', check_same_thread=False) as conn:
            memory = SqliteSaver(conn)
            return graph_builder.compile(checkpointer=memory)