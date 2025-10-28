from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver    # type: ignore 
from langchain_core.messages import ToolMessage                  # type: ignore
import aiosqlite                                                     # type: ignore
from typing import List, Any
from agents.pdf2sql_agent.src.config import (
    AGENT_CHECKPOINT_DB,
    AGENT_VERBOSE
)
from agents.pdf2sql_agent.src.agent.agent_state import SQL2PDFAgentState
from langgraph.prebuilt import ToolNode



from agents.pdf2sql_agent.src.agent.prompts import get_system_prompt
from agents.pdf2sql_agent.src.agent.tools import (
    execute_sql_query,
    generate_pdf_report,
    execute_visualization_tool,
)
from agents.pdf2sql_agent.src.agent.tools.memory_tool import memory_search_tool
from agents.pdf2sql_agent.src.memory.qdrant_store import QdrantStore
from agents.pdf2sql_agent.src.memory.memory_manager import ClaudeMemoryManager


class SQL2PDFAgent:

    def __init__(self, name: str = "SQL2PDFAgent", description: str = None, enable_critique: bool = False, enable_memory: bool = False, tools: List[Any] | None = None):
        """
        Initialize the SQL2PDFAgent

        Args:
            name: Name of the agent
            description: Custom description (defaults to system prompt)
            enable_critique: Whether to enable the critique/self-review loop
            enable_memory: Whether to enable long term memory (needs qdrant running)
        """
        self.name = name
        self.description = description or get_system_prompt()
        self.enable_critique = enable_critique
        self.enable_memory = enable_memory
        self.llm = init_chat_model("gpt-4o-mini")
        # Initialize Claude LLM
        """
        I don't have access to this
        self.llm = ChatAnthropic(
            model=AGENT_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=AGENT_TEMPERATURE,
        )
        """


        # Define available tools
        self.tools = list(tools) if tools else [execute_sql_query, generate_pdf_report, execute_visualization_tool]
        # add memory components if enabled
        if self.enable_memory:
            try:
                self.tools.append(memory_search_tool)
                self.memory_store = QdrantStore(collection_name="SQL2PDFAgent")
                self.memory_manager = ClaudeMemoryManager()
                if AGENT_VERBOSE:
                    print(f"[{self.name}] Memory enabled - {self.memory_store.count()} memories stored")
            except Exception as e:
                if AGENT_VERBOSE:
                    print(f"[{self.name}] Memory init failed: {e}, continuing without memory")
                self.enable_memory = False
                self.memory_store = None
                self.memory_manager = None
        else:
            self.memory_store = None
            self.memory_manager = None

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        if AGENT_VERBOSE:
            print(f"[{self.name}] Initialized with {len(self.tools)} tools")
            print(f"[{self.name}] Critique enabled: {self.enable_critique}")

    def retrieve_memory(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """retrieve relevant memories before planning"""
        if not self.enable_memory or not self.memory_store:
            return {}

        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Retrieving memories...")

        user_message = state["messages"][-1].content if state["messages"] else ""

        try:
            memories = self.memory_store.get(user_message, top_k=5, score_threshold=0.7)

            if memories:
                memory_context = "## past experiences:\n\n"
                for mem in memories:
                    if mem["memory_type"] == "Episode":
                        data = mem["memory_data"]
                        memory_context += f"- case: {data['observation']}\n  learned: {data['result']}\n\n"
                    elif mem["memory_type"] == "Semantic":
                        data = mem["memory_data"]
                        ctx = f" [{data['context']}]" if data.get('context') else ""
                        memory_context += f"- rule: {data['subject']} -> {data['predicate']} -> {data['object']}{ctx}\n\n"

                if AGENT_VERBOSE:
                    print(f"[{self.name}] Found {len(memories)} relevant memories")

                return {"retrieved_memories": memories, "memory_context": memory_context}

        except Exception as e:
            if AGENT_VERBOSE:
                print(f"[{self.name}] Memory retrieval error: {e}")

        return {}

    def planner(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """
        Create a step-by-step plan for handling the user's request

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Planning...")

        # Get the latest user message
        user_message = state["messages"][-1].content if state["messages"] else ""

        system_msg = (
            "You are an expert planner for a userprompt to SQL and PDF agent. "
            "Create a concise, step-by-step plan to answer the user's request. "
            "Consider which tools to use and in what order. "
            "If a message has come from another agent you must make sure that the contactotheragents tool is used to update the other agent"
            f"Available tools: {[tool.name for tool in self.tools]}\n"
        )

        # add memory context if available
        if state.get("memory_context"):
            system_msg += f"\n{state['memory_context']}"
            system_msg += "use this past knowledge when planning.\n"

        system_msg += "Respond with the plan only, as a numbered list."

        planner_messages = [("system", system_msg), ("user", user_message)]

        plan = self.llm.invoke(planner_messages).content

        if AGENT_VERBOSE:
            print(f"[{self.name}] Plan created:\n{plan}")

        return {"plan": plan}

    def chat(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """
        Main agent node that processes messages and calls tools

        Args:
            state: Current agent state

        Returns:
            Updated state with agent response
        """
        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Processing request...")

        # Build system prompt
        system_prompt = self.description

        # Add plan context if available
        if state.get("plan"):
            system_prompt += f"\n\nYour plan:\n{state['plan']}"

        # Add critique feedback if available (for revision)
        if state.get("critique") and state["critique"] != "None":
            system_prompt += f"\n\nYou must revise your previous answer based on this critique:\n{state['critique']}"
            if AGENT_VERBOSE:
                print(f"[{self.name}] Revising based on critique")

        # Prepare messages with system prompt
        messages_with_prompt = [("system", system_prompt)] + state["messages"]

        # Invoke LLM with tools
        ai_response = self.llm_with_tools.invoke(messages_with_prompt)

        if AGENT_VERBOSE:
            if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
                print(f"[{self.name}] Calling {len(ai_response.tool_calls)} tools: "
                      f"{[tc['name'] for tc in ai_response.tool_calls]}")
            else:
                print(f"[{self.name}] Responding to user (no tools called)")

        return {"messages": [ai_response]}

    def tools_node(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """
        Execute tools called by the agent

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        outputs = []
        last_message = state["messages"][-1]

        tools_by_name = {tool.name: tool for tool in self.tools}

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if AGENT_VERBOSE:
                    print(f"[Tool] Executing {tool_name}...")

                tool = tools_by_name[tool_name]
                try:
                    result = tool.invoke(tool_args)
                    if AGENT_VERBOSE:
                        print(f"[Tool] {tool_name} completed successfully")
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                    if AGENT_VERBOSE:
                        print(f"[Tool] {tool_name} failed: {str(e)}")

                # Create tool message
                tool_message = ToolMessage(
                    content=result,
                    name=tool_name,
                    tool_call_id=tool_id
                )
                outputs.append(tool_message)

        return {"messages": outputs}

    def critique(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """
        Review the agent's response for quality and completeness

        Args:
            state: Current agent state

        Returns:
            Updated state with critique feedback
        """
        if not self.enable_critique:
            # Skip critique if disabled
            return {"critique": "None"}

        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Critiquing response...")

        critique_prompt = (
            "You are an expert critic reviewing a response to a user's request about SQL to PDF reports.\n"
            "Evaluate if the answer is:\n"
            "- Complete: Does it address all parts of the user's query?\n"
            "- Accurate: Is the information correct?\n"
            "- Helpful: Does it provide actionable insights?\n\n"
            f"Original plan:\n{state.get('plan', 'No plan available')}\n\n"
            "If the response is good, respond with ONLY the word 'yes'.\n"
            "If it needs improvement, provide a brief critique (1-2 sentences) of what's missing or wrong."
        )

        critique_message = [
            state["messages"][-1],
            ("system", critique_prompt)
        ]

        critique = self.llm.invoke(critique_message).content

        if AGENT_VERBOSE:
            print(f"[{self.name}] Critique: {critique}")

        if "yes" in critique.lower():
            return {"critique": "None"}
        else:
            return {"critique": critique}

    def route_after_chat(self, state: SQL2PDFAgentState) -> Literal["tools", "critique", "end"]:
        """
        Decide whether to call tools, critique, or end after chat node

        Args:
            state: Current agent state

        Returns:
            Next node to visit
        """
        ai_message = state["messages"][-1]

        # If tools were called, go to tools node
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"

        # If critique is enabled, go to critique
        if self.enable_critique:
            return "critique"

        # Otherwise, end
        return "end"

    def route_after_critique(self, state: SQL2PDFAgentState) -> Literal["chat", "end"]:
        """
        Decide whether to revise (go back to chat) or end after critique

        Args:
            state: Current agent state

        Returns:
            Next node to visit
        """
        if state.get("critique") and state["critique"] != "None":
            # Needs revision
            return "chat"
        else:
            # Good to go
            return "end"

    def store_memory(self, state: SQL2PDFAgentState) -> SQL2PDFAgentState:
        """extract and store memories from conversation"""
        if not self.enable_memory or not self.memory_manager:
            return {}

        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Storing memories...")

        try:
            extracted = self.memory_manager.extract(state["messages"])

            if AGENT_VERBOSE:
                print(f"[{self.name}] Extracted {len(extracted)} memories from conversation")
                for i, mem in enumerate(extracted, 1):
                    mem_type = type(mem).__name__
                    if mem_type == "Episode":
                        print(f"  {i}. Episode: {mem.observation[:80]}...")
                    else:
                        print(f"  {i}. Semantic: {mem.subject} -> {mem.predicate} -> {mem.object}")

            stored_count = 0
            skipped_count = 0
            for memory in extracted:
                try:
                    mem_id = self.memory_store.put(memory, check_duplicates=True)
                    # Check if it was actually stored or was a duplicate
                    if mem_id:
                        stored_count += 1
                except Exception as e:
                    if AGENT_VERBOSE:
                        print(f"  Error storing memory: {e}")
                    skipped_count += 1

            if AGENT_VERBOSE:
                if stored_count > 0:
                    print(f"[{self.name}] Stored {stored_count} new memories")
                if skipped_count > 0:
                    print(f"[{self.name}] Skipped {skipped_count} memories (duplicates or errors)")

        except Exception as e:
            if AGENT_VERBOSE:
                print(f"[{self.name}] Memory storage error: {e}")

        return {}

    async def build_graph(self, connection) -> StateGraph:
        """
        Build and compile the agent graph with checkpointing

        Args:
            connection: aiosqlite database connection for checkpointing

        Returns:
            Compiled graph ready for execution
        """
        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Building agent graph...")

        # Create graph
        graph = StateGraph(SQL2PDFAgentState)

        tool_node = ToolNode(tools=self.tools)
        # Add nodes
        graph.add_node("planner", self.planner)
        graph.add_node("chat", self.chat)
        graph.add_node("tools", tool_node)

        if self.enable_critique:
            graph.add_node("critique", self.critique)

        if self.enable_memory:
            graph.add_node("retrieve_memory", self.retrieve_memory)
            graph.add_node("store_memory", self.store_memory)

        # Add edges
        if self.enable_memory:
            graph.add_edge(START, "retrieve_memory")
            graph.add_edge("retrieve_memory", "planner")
        else:
            graph.add_edge(START, "planner")

        graph.add_edge("planner", "chat")
        graph.add_edge("tools", "chat")

        # Conditional edges from chat
        if self.enable_critique:
            graph.add_conditional_edges(
                "chat",
                self.route_after_chat,
                {
                    "tools": "tools",
                    "critique": "critique",
                    "end": "store_memory" if self.enable_memory else END
                }
            )

            graph.add_conditional_edges(
                "critique",
                self.route_after_critique,
                {
                    "chat": "chat",
                    "end": "store_memory" if self.enable_memory else END
                }
            )
        else:
            graph.add_conditional_edges(
                "chat",
                self.route_after_chat,
                {
                    "tools": "tools",
                    "end": "store_memory" if self.enable_memory else END
                }
            )

        if self.enable_memory:
            graph.add_edge("store_memory", END)

        # Set up checkpointing
        memory = AsyncSqliteSaver(connection)

        if AGENT_VERBOSE:
            print(f"[{self.name}] Graph compiled with checkpointing")

        return graph.compile(checkpointer=memory)


# Convenience function for backward compatibility
async def create_agent_app(enable_critique: bool = False, enable_memory: bool = False):
    """
    Create and compile the SQL to PDF Report Agent

    Args:
        enable_critique: Whether to enable the critique/self-review loop
        enable_memory: Whether to enable long term memory (needs qdrant)

    Returns:
        Compiled agent graph ready for execution
    """
    #create database connection
    conn = await aiosqlite.connect(AGENT_CHECKPOINT_DB)

    #create agent instance
    agent = SQL2PDFAgent(
        name="SQL2PDFAgent",
        description=get_system_prompt(),
        enable_critique=enable_critique,
        enable_memory=enable_memory
    )


    graph = await agent.build_graph(conn)

    await graph.checkpointer.setup()

    if AGENT_VERBOSE:
        print(f"[Agent] Initialized with checkpointing to: {AGENT_CHECKPOINT_DB}")
        if enable_memory:
            print(f"[Agent] Long term memory is ENABLED")

    return graph


# Legacy function for backward compatibility
async def build_agent_graph() -> StateGraph:
    """
    Legacy function - builds agent graph without compilation
    Kept for backward compatibility
    """
    from agents.pdf2sql_agent.src.agent.agent_state import SQL2PDFAgentState

    graph = StateGraph(SQL2PDFAgentState)

    agent = SQL2PDFAgent()

    graph.add_node("planner", agent.planner)
    graph.add_node("chat", agent.chat)
    graph.add_node("tools", agent.tools_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "chat")
    graph.add_edge("tools", "chat")

    graph.add_conditional_edges(
        "chat",
        agent.route_after_chat,
        {
            "tools": "tools",
            "end": END
        }
    )

    return graph
