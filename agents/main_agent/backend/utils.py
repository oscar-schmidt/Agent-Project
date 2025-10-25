import asyncio
from functools import wraps
from copy import deepcopy
import json
import re
import os
from typing import Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import wraps
from transformers import pipeline
from agents.main_agent.backend.model.states.StateManager import StateManager

from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState

load_dotenv()

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
summary_pipeline = pipeline("summarization", model=SUMMARIZER_MODEL)


def log_decorator(function):
    if asyncio.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            state = kwargs.get("state") or getattr(
                st.session_state, "state", None)
            if state is None:
                state = StateManager.get_state()
            try:
                if hasattr(state, "logs") and state.logs is not None:
                    state.logs.append(f"[{function.__name__}] called")
                result = await function(*args, **kwargs)
                return result
            except Exception as e:
                state.logs.append(f"[{function.__name__}] ERROR. {e}")
                raise
        return async_wrapper
    else:
        @wraps(function)
        def sync_wrapper(*args, **kwargs):
            state = kwargs.get("state") or getattr(
                st.session_state, "state", None)
            if state is None:
                state = StateManager.get_state()
            try:
                if hasattr(state, "logs") and state.logs is not None:
                    state.logs.append(f"[{function.__name__}] called")
                return function(*args, **kwargs)
            except Exception as e:
                state.logs.append(f"[{function.__name__}] ERROR. {e}")
                raise
        return sync_wrapper


@log_decorator
def get_chunk(data: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(data)


@log_decorator
def get_embedding(chunk: str):
    if isinstance(chunk, str):
        chunk = [chunk]
    embed_model = load_model()
    embed_result = embed_model.encode(
        chunk, show_progress_bar=True, normalize_embeddings=True)
    return normalize_vector(embed_result)


def normalize_vector(vec: list[float]) -> list[float]:
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


#@st.cache_resource
def load_model():
    return SentenceTransformer(model_name_or_path=EMBED_MODEL)


@log_decorator
def single_chunk_summary(single_chunk: str,  min_len: Optional[int] = None, max_len: Optional[int] = None) -> str:
    single_chunk_len = len(single_chunk.split())
    max_new = max_len if max_len else max(1, single_chunk_len // 2)
    min_len = min_len if min_len else min(50, single_chunk_len)
    summary = summary_pipeline(
        single_chunk,
        max_new_tokens=max_new,
        min_length=min_len,
        do_sample=False
    )
    if summary and "summary_text" in summary[0]:
        return summary[0]["summary_text"].replace("\xa0", " ")
    return ""


@log_decorator
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"•+", "-", text)
    text = text.strip()
    return text


def get_user_input() -> str:
    state = StateManager.get_state()
    user_input = state.messages.user_query_list[-1].content if state.messages.user_query_list else None
    return user_input
