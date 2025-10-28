import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.StateManager import StateManager
from constants import SYSTEM_PROMPT_LIST
from langchain.chat_models import init_chat_model # update model to use langcahin prebuilt chat model init function
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode
from common.tools.date_time import DateTime
class Agent:
    def __init__(self, name: str, prompt: str, tools: list):
        self.state = StateManager().get_state()
        self.name = name
        self.prompt = prompt
        self.llm = init_chat_model("ollama:qwen3:8b")  # updated this for ollama as "ollama:model_name"
        self.tools = tools
        #self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.llm_openai = init_chat_model("gpt-4o-mini")
        self.llm_openai_tools = self.llm_openai.bind_tools(self.tools)
        """
        Maybe use this later
        
        self.manager = create_memory_manager(
            "gpt-4o-mini",
            schemas=[Episode, Semantic],
            instructions="Extract all user information and events as Episodes, and any facts as Semantic",
        )
        """


    def planner(self, state: GraphState) -> GraphState:
        user_input = state.messages[-1].content
        planner_messages =  [("user", f"{user_input}")] + [
            (
                "system",
                "You are an expert planner, create a concise, step-by-step plan to answer the user's request. Respond with the plan only"
                f"These are the tools available for use {self.tools}"
            )
        ] + state.messages
        plan = self.llm_openai_tools.invoke(planner_messages).content
        print(f"Generated Plan: {plan}")
        state.plan  = plan

        return state

    def chat_agent(self, state: GraphState):

        system_prompt = SYSTEM_PROMPT_LIST.default_prompt + f"\nHere is the plan to follow:\n{state.plan}"
        print(f"System Prompt: {system_prompt}")

        response = self.llm_openai_tools.invoke(
            [{"role": "system", "content": system_prompt}] + state.messages
        )
        return {"messages":[response]}

    def route_tools(self, state: GraphState):
        ai_message = state.messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END



    async def graph_builder(self, connection):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self.planner)
        graph.add_node("chatbot", self.chat_agent)
        tool_node = ToolNode(tools=self.tools)
        graph.add_node("tools", tool_node)


        graph.add_edge(START, "planner")
        graph.add_edge("planner", "chatbot")
        graph.add_edge("tools", "chatbot")
        graph.add_conditional_edges(
            "chatbot",
            self.route_tools,
            {"tools": "tools", "__end__": END},
        )

        memory = AsyncSqliteSaver(connection)
        return graph.compile(checkpointer=memory)

datetime = DateTime()
agent = Agent(
    name="MainAgent",
    prompt=SYSTEM_PROMPT_LIST.default_prompt,
    tools=[datetime],)

async def main():
    connection = await aiosqlite.connect(f'db/test.db')
    graph = await agent.graph_builder(connection)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit" or user_input == "quit":
            break
        text_to_send = {"messages": [HumanMessage(content=user_input )]}
        finale_state = await graph.ainvoke(text_to_send, config={"thread_id": "123"})
        print(f"{finale_state['messages'][-1].content}")



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())