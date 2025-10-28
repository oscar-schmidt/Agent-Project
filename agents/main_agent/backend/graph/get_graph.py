import time
import aiosqlite
from typing import Any, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import os
from langchain.chat_models import init_chat_model
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.tools.bind_tool.finalized_tool import finalized_tool
#from agents.main_agent.backend.tools.command import command
from agents.main_agent.backend.tools.critique import critique
from agents.main_agent.backend.utils import log_decorator
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
load_dotenv()

SQL_PATH = os.getenv("SQL_PATH")

_memory_instance = None
_compiled_graph = None


async def get_memory():
    global _memory_instance
    if _memory_instance is None:
        conn = await aiosqlite.connect(SQL_PATH)
        _memory_instance = AsyncSqliteSaver(conn)
        await _memory_instance.setup()
    return _memory_instance




class MainAgent:
    def __init__(self, tools: List[Any]):
        #self.tool_registry = get_tool_registry(websocket)
        #self.tool_name_args = get_tool_name_list
        self.llm = init_chat_model("gpt-4o-mini")
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    @log_decorator
    def get_bind_tools(self, state: GraphState):

        tool_to_bind = []

        for tool_info in self.tool_registry.values():
            if tool_info["condition"](state, self.user_input):
                tool_to_bind.append(tool_info)

        tool_to_bind.sort(key=lambda t: t["priority"])

        state.logs.append(
            f"bind_tools: {[bind_tool['name'] for bind_tool in tool_to_bind]}")

        return tool_to_bind

    async def agent_chat(self, state: GraphState):
        default_prompt = (
            """
            You are the "Business Insights Agent," a specialist AI.
            
            ### Primary Goal
            Your primary goal is to help businesses improve by getting reviews and anlaysis from other agents. 
            You will classify reviews, perform sentiment analysis, and use this data 
            (along with internal documents) to provide actionable, strategic recommendations.
            
            ### Core Responsibilities
            1.  **Recommend Improvements:** Use your analysis to suggest specific, actionable ways the 
            business can improve its services, products, or operations.
            3.  **Consult Knowledge Base:** You have access to an internal knowledge base of company 
            documents and business improvement guides. You can summarize and draw insights from these 
            documents to support your recommendations.
            
            ### How to Respond
            You are a conversational assistant. Your task is to respond directly to the user's requests 
            in a helpful, natural, and informative way. Analyze the user's needs and use your knowledge 
            base and analytical skills to provide a complete answer.
            
            ### Example Workflow
            1.  **User:** "Here are last month's customer reviews: [Data...]. We need to know why our 
            sales are down this quarter."
            2.  **Your Thought:** "I have been given the reviews. I also have access to the company's 
            internal Q3 performance report in my knowledge base. I will analyze the sentiment and key 
            topics in these reviews and then cross-reference them with the performance report to find correlations."
            3.  **Your Response (Conversational):**
                "Thank you for providing the reviews. I have analyzed them and cross-referenced the 
                findings with the Q3 performance report from my knowledge base. My analysis shows that 
                'shipping delays' were mentioned in 42% of negative reviews, which directly correlates 
                with the customer churn noted in the Q3 report. I recommend we prioritize a review of our 
                logistics partner.
                """
        )
        print(default_prompt)
        messages_with_prompt = [("system", default_prompt)] + state.messages
        response = self.llm_with_tools.invoke(messages_with_prompt)
        return {"messages":[response]}



    @log_decorator
    async def invoke_tool(self, state: GraphState):

        time.sleep(10)
        ai_message = state.messages[-1]
        print(ai_message)
        tool_name_args = self.tool_name_args(state=state, message=state.messages[-1])
        bind_tools = self.get_bind_tools(state=state)
        if not bind_tools or len(bind_tools) == 0 or not tool_name_args:
            state.messages.append(
                AIMessage(content="No tools available. Responding directly."))
            return

        max_attempts = 2
        attempt = 0
        should_recall = True

        while attempt < max_attempts and should_recall:
            for tool_name, args in tool_name_args:
                new_state = await self.execute_tool(tool_name, args, state)

            finalized_result = await finalized_tool().ainvoke({"state": state})
            new_state = command("chat_tool", finalized_result)

            should_recall = critique(
                self.user_input, finalized_result.agent_response)

            attempt += 1

        finalized_result = await finalized_tool().ainvoke({"state": state})
        final_state = command("chat_tool", finalized_result)

    async def execute_tool(self, tool_name, args, state: GraphState) -> GraphState:
        bind_tools = self.get_bind_tools(state=state)
        tool_to_invoke = next(
            (t for t in bind_tools if t["name"] == tool_name), None)

        if tool_to_invoke is None:
            state.logs.append(f"[execute_tool] Tool {tool_name} not found")
            return state

        if "tools" in args and isinstance(args["tools"], list):
            nested_tools = args["tools"]
            for nested in nested_tools:
                nested_name = nested.get("tool_name")
                if nested_name == tool_name:
                    state.logs.append(
                        f"[execute_tool] Skipped self-nesting: {tool_name}")
                    continue
                nested_args = nested.get("args", {})
                nested_args["state"] = state
                state = self.execute_tool(nested_name, nested_args, state)

        result = await tool_to_invoke["invoke"].ainvoke(args)
        new_state = command(tool_name, result)

        return new_state

    def route_tools(self, state: GraphState):
        ai_message = state.messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "invoke_tool"
        return "__end__"

    async def get_graph(self, connection):
        graph = StateGraph(GraphState)

        graph.add_edge(START, "agent_chat")
        graph.add_node("agent_chat", self.agent_chat)
        tool_node = ToolNode(tools=self.tools)
        graph.add_node("tools", tool_node)

        graph.add_conditional_edges(
            "agent_chat",
            self.route_tools,
            {"tools": "tools", "__end__": END},
        )
        graph.add_edge("tools", "agent_chat")


        memory = AsyncSqliteSaver(connection)
        return graph.compile(checkpointer=memory)

async def main():
    agent = MainAgent(websocket=None)
    connection = await aiosqlite.connect(f'db/test.db')
    graph = await agent.get_graph(connection=connection)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit" or user_input == "quit":
            break
        text_to_send = {"messages": [HumanMessage(content=user_input)]}
        finale_state = await graph.ainvoke(text_to_send, config={"thread_id": "123"})
        print(f"{finale_state['messages'][-1].content}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())