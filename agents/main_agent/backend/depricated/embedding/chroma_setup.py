from datetime import datetime
import os
import hashlib
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from agents.main_agent.backend.model.states.StateManager import StateManager
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from agents.main_agent.backend.model.states.qa_state.DocTextClass import Meta
from agents.main_agent.backend.utils import get_embedding, log_decorator

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")
PDF_SUMMARY_COLLECTION = os.getenv("PDF_SUMMARY_COLLECTION")


@log_decorator
def get_or_create_doc_collection():
    state = StateManager.get_state()

    doc_name = state.qa_state.doc_name

    chroma_client = PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        name=doc_name,
        metadata={
            "description": f"pdf {doc_name} chunks and chunk summary",
            "created": str(datetime.now()),
            "distance_metric": "cosine"
        })
    state.logs.append(
        f"[collection] {doc_name} created or found")
    return collection


@log_decorator
def get_all_collection_name(state: GraphState):
    client = PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    collection_names_list = [c.name for c in collections]
    state.collection_names_list = collection_names_list


@log_decorator
def get_or_create_summary_collection(state: GraphState):

    chroma_client = PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        name=PDF_SUMMARY_COLLECTION,
        metadata={
            "description": f"pdf {PDF_SUMMARY_COLLECTION} file summary",
            "created": str(datetime.now()),
            "distance_metric": "cosine"
        })
    state.logs.append(
        f"[collection] {PDF_SUMMARY_COLLECTION} created or found")
    return collection


@log_decorator
def get_collection(collection_name: str):
    chroma_client = PersistentClient(path=CHROMA_PATH)
    try:
        collection = chroma_client.get_collection(name=collection_name)
        if collection and collection.count():
            return collection
        return None
    except Exception:
        return None


@log_decorator
def insert_data_row(data: str, embedding: list[float], metadata: Meta):
    collection = get_or_create_doc_collection()

    # total_chunk = len(state.qa_state.chunked_doc_text)

    # for i, pdf_text in enumerate(state.qa_state.chunked_doc_text, start=0):

    #     if pdf_text.embedding is None:
    #         state.logs.append(f"Skipping chunk {i}: no embedding found.")
    #         continue

    #     exist_chunk = collection.get(ids=[chunk_id(pdf_text.chunk)])
    #     if exist_chunk["documents"]:
    #         state.logs.append(f"Skipping chunk {i}: chunk already embedded.")
    #         continue

    collection.add(
        ids=[chunk_id(data)],
        embeddings=embedding,
        documents=[data],
        metadatas=[metadata]
    )

    # pdf_text.embedding = None

    #     state.logs.append(
    #         f"Inserted {i+1}/{total_chunk} chunk into Chroma.")


@log_decorator
def insert_pdf_summary(state: GraphState) -> GraphState:

    if state.summary_state.final_summary:
        return state

    chroma_client = PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        name=PDF_SUMMARY_COLLECTION,
        metadata={
            "description": "user upload pdf file summary",
            "created": str(datetime.now()),
            "doc_name": PDF_SUMMARY_COLLECTION,
            "distance_metric": "cosine"
        })

    exist_chunk = collection.get(
        ids=[chunk_id(state.summary_state.final_summary)])
    if exist_chunk and exist_chunk["documents"]:
        state.logs.append(f"Skipping final_summary: already embedded.")
        return state

    summary_embedding = get_embedding([state.summary_state.final_summary])

    collection.add(
        ids=[chunk_id(state.summary_state.final_summary)],
        embeddings=summary_embedding.tolist(),
        documents=[state.summary_state.final_summary],
        metadatas=[{"doc_name": PDF_SUMMARY_COLLECTION}]
    )

    state.logs.append(
        f"Inserted final_summary: {state.summary_state.final_summary}.")

    return state


def chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
