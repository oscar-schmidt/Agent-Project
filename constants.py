from types import SimpleNamespace

SYSTEM_MESSAGE_LIST = SimpleNamespace(
    top_k_kb_found_prompt=(
        "User asked: {user_input}. Using the knowledge below, "
        "please provide an exact answer if available. "
        "Knowledge: {top_k_kb}"
    ),
    top_k_kb_not_found_prompt=("There is no knowledge base available,"
                               "reply user 'No KB Found'"),

)


SYSTEM_LOG_LIST = SimpleNamespace(
    rag_router_log_template=SimpleNamespace(
        get_collection_exception=(
            "[rag_router] go to chat_agent. get_or_create_summary_collection error: {e}"),
        empty_collection_error=(
            "[rag_router] PDF_SUMMARY_COLLECTION is empty, go to [chat_agent]"),
        rag_retrieval_exception=(
            "[rag_router] [get_embedding] [rag_retrieval_node] error: {e}"),
        no_user_input_error=("[rag_router] No user_input, go to [chat_aget]"),
        successful_log=(
            "[rag_router] go to[rag_agent] top_score: {top_score}; rag_threshold: {RAG_THRESHOLD}"),
        no_kb_log=(
            "[rag_router] There is no revelant kb. Go to [no_answer_agent],top_score: {top_score}; rag_threshold: {RAG_THRESHOLD}"),
        chat_agent_log=(
            "[rag_router] top_score: {top_score} <= 0, go to chat_agent ")
    )
)

SYSTEM_PROMPT_LIST = SimpleNamespace(
    default_prompt="You are a helpful assistant.",
    tool_router_prompt=(
        "You are a Tool Router.\n"
        "Available tools: {tool_names}\n"
        "Decide which tool to use based on the user input. user_input: {user_input}\n"
        "You can choose to call only one tool or both tools if relevant.\n"
        "Use the tool by calling it directly; do not just return JSON.\n"
        "If no tool is needed, respond naturally to the user.\n"
        "- Respond ONLY with a valid JSON object in this exact format:\n"
        "  {{'tools': [\n"
        "      {{'tool_name': '<tool_name>', 'args': {{}}}},\n"
        "      ...\n"
        "  ]}}\n"
        "- Do not include any extra text or explanation.\n"
        "- JSON must be parseable.\n"
    ),
    final_summary_prompt=(
        "You are an AI tasked with reading any given text input, which could be a user feedback summary, "
        "PDF content, article, story, novel, report, or any other form of text.\n\n"

        "Available final summary reference: {final_summary}\n"

        "Your task is to analyze the text and produce ONE unified, concise, and insightful summary or overview.\n"
        "You may choose to call one, both, or neither of the tools as needed.\n"
        "Ensure that the final output captures the core information, key themes, and important insights.\n\n"

        "=== GUIDELINES ===\n"
        "1. Identify and focus on the most important points, recurring themes, or patterns in the text.\n"
        "2. Organize related ideas together clearly.\n"
        "3. Avoid repeating points or including unnecessary details.\n"
        "4. Use a format that suits the content: natural paragraphs, numbered lists, or bullet points are all acceptable.\n"
        "5. The summary should be clear, concise, and easy to understand, suitable for someone to grasp the main points quickly.\n"

        "Only include the final summary or overview.\n"
        "Do NOT include your reasoning process, any metadata, or JSON.\n"
        "Your response should directly reflect the key insights of the input text."
    ),
    chat_tool_prompt=(
        "You are an answer finalizer agent. "
        "Based on the user's input and the following tool outputs, "
        "compose a final polished answer.\n\n"
        "=== Tool Outputs ===\n{final_tool_output}\n\n"
        "=== Task ===\nGenerate the best final answer for the user."
    ),
    # chat_tool_prompt=(
    #     "You are the final reasoning agent.\n"
    #     "You will receive outputs from multiple tools. Please:\n"
    #     "1. Present each tool's result in the same order they appear.\n"
    #     "2. Then give a natural, concise final answer combining all tool results.\n\n"
    #     "Tool Outputs:\n{final_tool_output}"
    # )


)

# In residential care,how many resident deaths in 2022–23
# how many First Nations people using home support were aged under 65
# what is home care
