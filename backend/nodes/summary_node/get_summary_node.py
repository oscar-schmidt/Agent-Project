import copy
import os
from dotenv import load_dotenv

from backend.dataBase_setup.chroma_setup import get_all_collection_name, get_collection, insert_pdf_summary
from backend.model.states.graph_state.GraphState import GraphState
from backend.utils import get_embedding, get_user_input, single_chunk_summary, clean_text, log_decorator

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
SUMMARY_MIN_LENGTH = int(os.getenv("SUMMARY_MIN_LENGTH", 1000))
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", 2000))


def get_summary_node(state: GraphState) -> GraphState:
    user_input = get_user_input()
    collectin_name = get_collection_name(user_input)
    if collectin_name:
        chunked_text_content = format_chunk_list(collectin_name)
    else:
        chunked_text_content = [
            copy.deepcopy(chunk.chunk) for chunk in state.qa_state.chunked_doc_text
        ]

    if not state.summary_state.chunk_summary or len(state.summary_state.chunk_summary) < len(chunked_text_content):
        state.summary_state.chunk_summary = []
        for idx, chunk_content in enumerate(chunked_text_content, start=1):
            state.logs.append(
                f"[Summary Node] Processing chunk {idx}/{len(chunked_text_content)}"
            )
            clean_chunk = clean_text(chunk_content)
            chunk_summary = single_chunk_summary(clean_chunk)
            state.summary_state.chunk_summary.append(chunk_summary)

    if state.summary_state.chunk_summary:
        summaries_string = " ".join(state.summary_state.chunk_summary)
        input_len = len(summaries_string.split())
        max_len = min(SUMMARY_MAX_LENGTH, input_len)
        min_len = min(SUMMARY_MIN_LENGTH, max_len - 1)
        state.summary_state.final_summary = single_chunk_summary(
            summaries_string, min_len=min_len, max_len=max_len)

        insert_pdf_summary(state)
    else:
        state.summary_state.final_summary = "No content to summarize."

    return state


def get_collection_name(user_input: str) -> str | None:
    user_input = user_input.strip().lower()

    collection_names = get_all_collection_name()

    for name in collection_names:
        if name.lower() in user_input:
            return name

    return None


def format_chunk_list(collection_name: str):

    collection = get_collection(collection_name)

    result = collection.get(include=["documents"])
    documents = result.get("documents", [])

    return documents
