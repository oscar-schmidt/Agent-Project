from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from matplotlib.textpath import text_to_path

from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.StateManager import StateManager
from agents.main_agent.backend.utils import get_user_input, log_decorator
from constants import SYSTEM_PROMPT_LIST
from langchain.chat_models import init_chat_model # update model to use langcahin prebuilt chat model init function
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode
from common.tools.date_time import DateTime
#from agents.main_agent.backend.tools.tool_invoke_agent import ToolNode
class Agent:
    def __init__(self, name: str, prompt: str, tools: list):
        self.state = StateManager().get_state()
        self.name = name
        self.prompt = prompt
        self.llm = init_chat_model("ollama:qwen3:8b")  # updated this for ollama as "ollama:model_name"
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
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
        print(state)
        user_input = state.messages[-1].content
        print(f"User Input: {user_input}")
        planner_messages =  [("user", f"{user_input}")] + [
            (
                "system",
                "You are an expert planner, create a concise, step-by-step plan to answer the user's request. Respond with the plan only"
                f"These are the tools available for use {self.tools}"
            )
        ]
        plan = self.llm_openai_tools.invoke(planner_messages).content
        print(f"Generated Plan: {plan}")
        state.plan  = plan

        return state

    def chat_agent(self, state: GraphState):
        user_input = state.messages[-1].content
        new_state = None
        system_prompt = SYSTEM_PROMPT_LIST.default_prompt + f"\nHere is the plan to follow:\n{state.plan}"
        print(f"User Input: {user_input}")
        print(f"System Prompt: {system_prompt}")
        if user_input:
            response = self.llm_openai_tools.invoke(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            )
            print(response)
            state.logs.append(system_prompt)
            state.logs.append(response)
            state.messages.append(
                AIMessage(content=response.content))
            new_state = StateManager.update_state(state)
        return new_state

    def route_tools(self, state: GraphState):
        try:
            last_log = state.logs.logs[-1]
        except (AttributeError, IndexError):
            print("No logs available to inspect.")
            return END

        tool_calls = getattr(last_log, "tool_calls", None)
        if tool_calls and len(tool_calls) > 0:
            print(f"Routing to tools... found {len(tool_calls)} tool call(s).")
            return "tools"

        print("Ending graph execution.")
        return END



    async def graph_builder(self, state: GraphState):
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


        #memory = AsyncSqliteSaver(connection)
        checkpointer = InMemorySaver()
        return graph.compile()

datetime = DateTime()
agent = Agent(
    name="MainAgent",
    prompt=SYSTEM_PROMPT_LIST.default_prompt,
    tools=[datetime],)

async def main():



    text = "What is the date time please?"
    state = StateManager.get_state()
    state.messages.append(HumanMessage(content=text))
    updated_state = StateManager.update_state(state)
    graph = await agent.graph_builder(updated_state)
    text_to_send = {"messages": [HumanMessage(content=text)]}
    finale_state = await graph.ainvoke(text_to_send, config={"thread_id": "123"})
    if isinstance(finale_state, dict):
        finale_state = GraphState(**finale_state)

    StateManager.update_state(finale_state)
    message = finale_state.logs.logs[-1].content
    print(message )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())