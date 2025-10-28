import os
from dotenv import load_dotenv

from agents.main_agent.backend.utils import get_embedding, get_user_input, log_decorator

from agents.main_agent.backend.dataBase_setup.chroma_setup import get_all_collection_name, get_collection
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.nodes.qa_node.rag_retrieval_node import rag_retrieval_node
from agents.main_agent.backend.utils import get_embedding, get_user_input
from constants import SYSTEM_LOG_LIST



load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", 0.15))
PDF_SUMMARY_COLLECTION = os.getenv("PDF_SUMMARY_COLLECTION")


def rag_router(state: GraphState) -> str:
    """Decide whether should use rag"""

    log_template = SYSTEM_LOG_LIST.rag_router_log_template

    user_input = get_user_input()
    if not user_input:
        state.logs.append(log_template.no_user_input_error)
        return "FALSE"

    top_score = 0.0
    top_k_kb = None

    try:
        collection_names_search_list = get_all_collection_name()

        for collection_name in collection_names_search_list:
            collection = get_collection(collection_name)
            state.logs.append(
                f"Searching Collection: {collection_name}")
            if collection:
                if collection.count() == 0:
                    state.logs.append(
                        log_template.empty_collection_error)
                    continue
                try:
                    embed_user_input = get_embedding([user_input])
                    try:
                        _, top_k_kb, top_score = rag_retrieval_node(
                            state, collection, embed_user_input)
                    except Exception as e:
                        e = e
                except Exception as e:
                    state.logs.append(
                        log_template.rag_retrieval_exception.format(e=e))
                    return "FALSE"
                if top_k_kb and top_score >= RAG_THRESHOLD:
                    state.qa_state.top_k_kb = top_k_kb
                    state.logs.append(log_template.successful_log.format(
                        top_score=top_score, RAG_THRESHOLD=RAG_THRESHOLD))
                    return "TRUE"
                else:
                    continue
            else:
                return "FALSE"
    except Exception as e:
        state.logs.append(
            log_template.get_collection_exception.format(e=e))
        return "FALSE"

    if top_score <= 0:
        state.logs.append(
            log_template.chat_agent_log.format(top_score=top_score))
        return "FALSE"

    state.logs.append(log_template.no_kb_log.format(
        top_score=top_score, RAG_THRESHOLD=RAG_THRESHOLD))
    return "NO_KB"
