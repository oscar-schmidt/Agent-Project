from typing import Any
import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
from common.memory.semantic import Semantic
from common.memory.episodic import Episode
from typing import Union
from .Embedding import Embedding


MemoryObject = Union[Semantic, Episode]

class QdrantStore():
    def __init__(self, collection_name: str):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        self.embedding = Embedding()

    def get(self, query: str) -> list[dict[str, Any]]:
        """retrieves memories given a query or a text input and returns a list of relevant memories"""
        vector = self.embedding.get_embedding(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            limit=5, #update this to get more relevant memories
            query=vector,
        ).points

        memories = []
        for result in search_result:
            memories.append(result.payload)
        return memories

    def put(self, memories):
        """Stores the memory, while also checking for duplicate memories"""
        for memory in memories:
            vector = self.embedding.get_embedding(memory)
            check = self._check_for_duplicates(vector=vector)
            if check is None:
                id = memory.id
            else:
                id = check
            vector = self.embedding.get_embedding(memory.content)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=id,
                        vector=vector,
                        payload=memory.content.model_dump(),
                    )
                ]
            )

    def _check_for_duplicates(self, vector) -> str | int |None :
        """Takes a memory object semantic or episodic memories and checks weather there are any similar
        memories already in the storage"""

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            limit=1,
            query=vector,
            score_threshold=0.90,
        ).points

        if search_result:
            return search_result[0].id
        else:
            return None




