from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver    # type: ignore 
from langchain_core.messages import ToolMessage                  # type: ignore
import aiosqlite                                                     # type: ignore
from typing import List, Any
from agents.classification_agent.src.config import (
    AGENT_CHECKPOINT_DB,
    AGENT_VERBOSE
)
from agents.classification_agent.src.agent.agent_state import ReviewAgentState
from langgraph.prebuilt import ToolNode



from agents.classification_agent.src.agent.prompts import get_system_prompt
from agents.classification_agent.src.agent.tools.memory_tool import memory_search_tool
from agents.classification_agent.src.memory.qdrant_store import QdrantStore
from agents.classification_agent.src.memory.memory_manager import MemoryManager
from agents.classification_agent.src.agent.tools import (
    classify_review_criticality,
    analyze_review_sentiment,
    log_reviews_to_notion,
    get_current_datetime
)
from config.config_helper import get_model_config

class ReviewAgent:
    """A Review Classification Agent class for analyzing customer reviews"""

    def __init__(self, name: str = "ReviewAgent", description: str = None, enable_critique: bool = False, enable_memory: bool = False, tools: List[Any] | None = None):
        """
        Initialize the Review Classification Agent

        Args:
            name: Name of the agent
            description: Custom description (defaults to system prompt)
            enable_critique: Whether to enable the critique/self-review loop
            enable_memory: Whether to enable long term memory (needs qdrant running)
        """
        model_config = get_model_config()
        self.llm = init_chat_model(f"{model_config["provider"]}:{model_config['model']}")
        self.name = name
        self.description = description or get_system_prompt()
        self.enable_critique = enable_critique
        self.enable_memory = enable_memory
        #.llm = init_chat_model("ollama:qwen3:8b")





        # Define available tools
        self.tools = tools if tools is not None else [
            classify_review_criticality,
            analyze_review_sentiment,
            log_reviews_to_notion,
            get_current_datetime
        ]
        # add memory components if enabled
        if self.enable_memory:
            try:
                self.tools.append(memory_search_tool)
                self.memory_store = QdrantStore(collection_name="ReviewAgent")
                self.memory_manager = MemoryManager()
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

    def retrieve_memory(self, state: ReviewAgentState) -> ReviewAgentState:
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

    def planner(self, state: ReviewAgentState) -> ReviewAgentState:
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
            "Create a numbered step-by-step plan. Each step = ONE tool call.\n"
            f"Available tools: {[tool.name for tool in self.tools]}\n\n"
            "For review classification requests:\n"
            "Step 1: ingest_review (if raw review text provided)\n"
            "Step 2: classify_review_criticality\n"
            "Step 3: analyze_review_sentiment\n"
            "Step 4: log_reviews_to_notion\n"
            "Step 5: ContactOtherAgents (recipient_id=<sender_name>, message=<summary>) - if request came from another agent\n\n"
            "CRITICAL: ContactOtherAgents is MANDATORY for agent messages. It's a separate tool call, not commentary.\n"
            "For courtesy messages ('Thank you'), create plan: '1. No action needed (courtesy message)'\n"
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

    def chat(self, state: ReviewAgentState) -> ReviewAgentState:
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

            # Check if ContactOtherAgents is in plan but hasn't been called yet
            if "ContactOtherAgents" in state["plan"]:
                # Check message history to see if ContactOtherAgents has been called
                contact_called = any(
                    hasattr(msg, "tool_calls") and
                    any(tc.get("name") == "ContactOtherAgents" for tc in msg.tool_calls)
                    for msg in state["messages"]
                    if hasattr(msg, "tool_calls")
                )

                if not contact_called:
                    system_prompt += "\n\n⚠️ CRITICAL: Your plan includes ContactOtherAgents but you have NOT called it yet. You MUST call ContactOtherAgents as a tool - do NOT output text instead."

        # Add critique feedback if available (for revision)
        if state.get("critique") and state["critique"] != "None":
            system_prompt += f"\n\nYou must revise your previous answer based on this critique:\n{state['critique']}"
            if AGENT_VERBOSE:
                print(f"[{self.name}] Revising based on critique")

        # Prepare messages with system prompt
        messages_with_prompt = [("system", system_prompt)] + state["messages"]

        # Check what tools have been called in THIS turn (since the last user message)
        tools_called = []
        # Find the last user/human message index
        last_user_msg_idx = -1
        for i in range(len(state["messages"]) - 1, -1, -1):
            msg = state["messages"][i]
            if hasattr(msg, "type") and msg.type in ("human", "user"):
                last_user_msg_idx = i
                break
            # Check for HumanMessage class
            if msg.__class__.__name__ in ("HumanMessage", "UserMessage"):
                last_user_msg_idx = i
                break

        # Only count tool calls AFTER the last user message
        if last_user_msg_idx >= 0:
            for msg in state["messages"][last_user_msg_idx:]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_called.extend([tc.get("name") for tc in msg.tool_calls])

        # Enforce ContactOtherAgents if it's in the plan but hasn't been called
        if state.get("plan") and "ContactOtherAgents" in state["plan"]:
            # Check which required tools are missing
            has_classify = "classify_review_criticality" in tools_called
            has_sentiment = "analyze_review_sentiment" in tools_called
            has_logged = "log_reviews_to_notion" in tools_called
            has_replied = "ContactOtherAgents" in tools_called

            if AGENT_VERBOSE:
                print(f"[{self.name}] Enforcement check - classify:{has_classify}, sentiment:{has_sentiment}, logged:{has_logged}, replied:{has_replied}")

            # Step 2: Enforce classify_review_criticality if it's in plan but not called
            if "classify_review_criticality" in state["plan"] and not has_classify:
                # Put enforcement at the TOP to override everything
                enforcement_msg = "IMMEDIATE INSTRUCTION: Call classify_review_criticality tool NOW. Do not call any other tool.\n\n" + system_prompt
                system_prompt = enforcement_msg
                if AGENT_VERBOSE:
                    print(f"[{self.name}] >> ENFORCING step 2: classify_review_criticality")
            # Step 3: Enforce sentiment if classify done but sentiment not done
            elif "analyze_review_sentiment" in state["plan"] and has_classify and not has_sentiment:
                system_prompt = "IMMEDIATE INSTRUCTION: Call analyze_review_sentiment tool NOW. Do not call any other tool.\n\n" + system_prompt
                if AGENT_VERBOSE:
                    print(f"[{self.name}] >> ENFORCING step 3: analyze_review_sentiment")
            # Step 4: Enforce logging if both classify and sentiment done but logging not done
            # ALWAYS require logging - don't check if it's in the plan
            elif has_classify and has_sentiment and not has_logged and not has_replied:
                enforcement_msg = "!!!CRITICAL MANDATORY INSTRUCTION!!!\n\nYou MUST call log_reviews_to_notion tool RIGHT NOW.\nDo NOT call ContactOtherAgents.\nDo NOT call any other tool.\nDo NOT respond with text.\nONLY call log_reviews_to_notion with the merged JSON from classify and sentiment results.\n\nThis is NOT negotiable. Call log_reviews_to_notion NOW.\n\n" + system_prompt
                system_prompt = enforcement_msg
                if AGENT_VERBOSE:
                    print(f"[{self.name}] >> ENFORCING step 4: log_reviews_to_notion (REQUIRED - BLOCKING ContactOtherAgents)")
            # Step 5: Enforce ContactOtherAgents if all other steps done
            elif not has_replied:
                # Check if all required steps are complete
                classify_done = has_classify
                sentiment_done = has_sentiment
                logging_done = has_logged  # MUST actually be logged, not just skipped from plan

                if classify_done and sentiment_done and logging_done:
                    system_prompt = "STOP. You have completed all data processing. You MUST now call the ContactOtherAgents TOOL to reply to the sender. DO NOT write a summary. DO NOT output text. ONLY call the ContactOtherAgents tool with recipient_id and message parameters. This is MANDATORY.\n\n" + system_prompt
                    if AGENT_VERBOSE:
                        print(f"[{self.name}] >> ENFORCING step 5: ContactOtherAgents")

        # Dynamically filter tools based on workflow state
        # If logging hasn't been done but classify and sentiment are done, ONLY allow log_reviews_to_notion
        tools_to_use = self.llm_with_tools
        if state.get("plan") and "ContactOtherAgents" in state["plan"]:
            has_classify = "classify_review_criticality" in tools_called
            has_sentiment = "analyze_review_sentiment" in tools_called
            has_logged = "log_reviews_to_notion" in tools_called
            has_replied = "ContactOtherAgents" in tools_called

            if has_classify and has_sentiment and not has_logged and not has_replied:
                # Filter tools to ONLY log_reviews_to_notion
                filtered_tools = [t for t in self.tools if t.name == "log_reviews_to_notion"]
                tools_to_use = self.llm.bind_tools(filtered_tools)
                if AGENT_VERBOSE:
                    print(f"[{self.name}] >> FILTERING tools to ONLY: log_reviews_to_notion")

        # Invoke LLM with (possibly filtered) tools
        ai_response = tools_to_use.invoke(messages_with_prompt)

        if AGENT_VERBOSE:
            if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
                print(f"[{self.name}] Calling {len(ai_response.tool_calls)} tools: "
                      f"{[tc['name'] for tc in ai_response.tool_calls]}")
            else:
                print(f"[{self.name}] Responding to user (no tools called)")

        return {"messages": [ai_response]}

    def tools_node(self, state: ReviewAgentState) -> ReviewAgentState:
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

    def critique(self, state: ReviewAgentState) -> ReviewAgentState:
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
            "You are an expert critic reviewing a response to a user's request about review classification.\n"
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

    def route_after_chat(self, state: ReviewAgentState) -> Literal["tools", "critique", "end"]:
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

    def route_after_critique(self, state: ReviewAgentState) -> Literal["chat", "end"]:
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

    def store_memory(self, state: ReviewAgentState) -> ReviewAgentState:
        """extract and store memories from conversation"""
        if not self.enable_memory or not self.memory_manager:
            return {}

        if AGENT_VERBOSE:
            print(f"\n[{self.name}] Storing memories...")

        try:
            extracted = self.memory_manager.extract(state["messages"])

            stored_count = 0
            for memory in extracted:
                try:
                    self.memory_store.put(memory, check_duplicates=True)
                    stored_count += 1
                except:
                    pass

            if AGENT_VERBOSE and stored_count > 0:
                print(f"[{self.name}] Stored {stored_count} new memories")

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
        graph = StateGraph(ReviewAgentState)

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
    Create and compile the review classification agent

    Args:
        enable_critique: Whether to enable the critique/self-review loop
        enable_memory: Whether to enable long term memory (needs qdrant)

    Returns:
        Compiled agent graph ready for execution
    """
    #create database connection
    conn = await aiosqlite.connect(AGENT_CHECKPOINT_DB)

    #create agent instance
    agent = ReviewAgent(
        name="ReviewClassificationAgent",
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
    from agents.classification_agent.src.agent.agent_state import ReviewAgentState

    graph = StateGraph(ReviewAgentState)

    agent = ReviewAgent()

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
