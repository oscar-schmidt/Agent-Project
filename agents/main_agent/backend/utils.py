import asyncio
from functools import lru_cache, wraps
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
from langchain_core.messages import HumanMessage
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
summary_pipeline = pipeline("summarization", model=SUMMARIZER_MODEL)
provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if provider == "openai":
    client = OpenAI(api_key=OPENAI_API_KEY)


def log_decorator(function):
    if asyncio.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            state = kwargs.get("state")
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
            state = kwargs.get("state")
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


def get_chunk(data: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(data)


def get_embedding(chunk: str):
    if isinstance(chunk, str):
        chunk = [chunk]
    if provider == "openai":
        response = client.embeddings.create(
            model=os.getenv("OPENAI_EMBED_MODEL"),
            input=chunk
        )
        return [data.embedding for data in response.data]
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


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(model_name_or_path=EMBED_MODEL)


def single_chunk_summary(single_chunk: str,  min_len: Optional[int] = None, max_len: Optional[int] = None) -> str:
    if provider == "openai":
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_LIST.summary_prompt},
                {"role": "user", "content": single_chunk}
            ]
        )
        return resp.choices[0].message.content.strip()

    elif provider == "ollama":
        single_chunk_len = len(single_chunk.split())
        max_new = max_len if max_len else max(1, single_chunk_len // 2)
        min_new = min_len if min_len else min(50, single_chunk_len)
        if min_new > max_new:
            min_new = max_new // 2
        summary = summary_pipeline(
            single_chunk,
            max_new_tokens=max_new,
            min_length=min_new,
            do_sample=False
        )
        if summary and "summary_text" in summary[0]:
            return summary[0]["summary_text"].replace("\xa0", " ")
        return ""


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"•+", "-", text)
    text = text.strip()
    return text


def sanitize_doc_name(doc_name: str) -> str:
    doc_name = doc_name.strip()

    base_name, ext = os.path.splitext(doc_name)
    ext = ext.lower()

    base_name = re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)

    base_name = re.sub(r'^[^a-zA-Z0-9]+', '', base_name)
    base_name = re.sub(r'[^a-zA-Z0-9]+$', '', base_name)

    if len(base_name + ext) < 3:
        base_name = f"doc_{base_name}"
    elif len(base_name + ext) > 512:
        base_name = base_name[:512 - len(ext)]

    return base_name + ext


def get_user_input() -> str:
    state = StateManager.get_state()

    user_messages = [
        msg for msg in state.messages if isinstance(msg, HumanMessage)]

    if user_messages:
        return user_messages[-1].content
    return ""
