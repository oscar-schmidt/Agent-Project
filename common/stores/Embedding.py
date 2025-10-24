from typing import Any
import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
from common.memory.semantic import Semantic
from common.memory.episodic import Episode
from typing import Union

MemoryObject = Union[Semantic, Episode]

class Embedding:
    """
    Embedding class for the custom qdrant class, it uses open AI embedding by default
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"

    def semantic_to_sentence(self, memory: Semantic):
        """converts semantic memory to sentence representation for embedding"""
        return f"{memory.subject} {memory.predicate} {memory.object}"


    def episodic_to_sentence(self, memory: Episode):
        """converts episodic memory to sentence representation for embedding"""
        return f"{memory.observation} {memory.thoughts} {memory.action} {memory.result}"

    def get_embedding(self, quary: MemoryObject | str) -> list[float]:
        """Get the embedding for the given memory using openai. expects a memory object which contains
        both semantic and episodic memories
        """
        if isinstance(quary, Semantic):
            text = self.semantic_to_sentence(quary)
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            # as vector
            embeddings = response.data[0].embedding
            if not embeddings or len(embeddings) == 0:
                msg = "No embeddings returned from OpenAI."
                raise ValueError(msg)

            return list(embeddings)

        elif isinstance(quary, Episode):
            text = self.episodic_to_sentence(quary)
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embeddings = response.data[0].embedding
            if not embeddings or len(embeddings) == 0:
                msg = "No embeddings returned from OpenAI."
                raise ValueError(msg)

            return list(embeddings)

        elif isinstance(quary, Semantic):
            response = self.client.embeddings.create(
                model=self.model,
                input=quary,
            )
            embeddings = response.data[0].embedding
            if not embeddings or len(embeddings) == 0:
                msg = "No embeddings returned from OpenAI."
                raise ValueError(msg)

            return list(embeddings)

        elif isinstance(quary, str):
            response = self.client.embeddings.create(
                model=self.model,
                input=quary,
            )
            embeddings = response.data[0].embedding
            if not embeddings or len(embeddings) == 0:
                msg = "No embeddings returned from OpenAI."
                raise ValueError(msg)
            return list(embeddings)
        else:
            raise Exception(f"Unknown memory type: {type(memory)}")