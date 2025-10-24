import chromadb
from chromadb.config import Settings
from uuid import uuid4
from typing import Union, List, Dict, Optional
from datetime import datetime
import os

from .schemas import Episode, Semantic
from .embedding import EmbeddingGenerator


class MemoryStore:
    """vector db for long term memory - uses chromadb (no docker needed)"""

    def __init__(
        self,
        collection_name: str = "ReviewAgent",
        persist_directory: str = "./memory_storage"
    ):
        self.collection_name = collection_name
        self.embedder = EmbeddingGenerator()

        # create persist dir if doesnt exist
        os.makedirs(persist_directory, exist_ok=True)

        # init chromadb client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "long term memory for review classification agent"}
        )

    def put(
        self,
        memory: Union[Episode, Semantic],
        metadata: Optional[Dict] = None,
        check_duplicates: bool = True
    ) -> str:
        """store a memory"""
        memory_text = self.embedder.memory_to_text(memory)
        embedding = self.embedder.generate(memory_text)

        # check for dupes - 90% similarity
        if check_duplicates:
            similar = self.get(memory_text, top_k=1, score_threshold=0.9)
            if similar:
                return similar[0]["id"]

        memory_id = str(uuid4())

        # prepare metadata
        meta = {
            "memory_type": type(memory).__name__,
            "text": memory_text,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }

        # store memory data as json in documents
        memory_doc = str(memory.model_dump())

        # add to collection
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[memory_doc],
            metadatas=[meta]
        )

        return memory_id

    def get(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """search memories by similarity"""
        query_embedding = self.embedder.generate_query_embedding(query)

        # query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # format results
        memories = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, mem_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0

                # chromadb uses squared l2 distance, normalize to 0-1 similarity
                # smaller distance = more similar
                # typical range: 0-4 for normalized vectors
                score = max(0, 1.0 - (distance / 4.0))

                # always include results, let caller filter by threshold if needed
                # or use lower default threshold since chromadb scoring is different
                if score >= max(0.3, score_threshold - 0.4):  # adjust threshold for chromadb
                    meta = results['metadatas'][0][i]
                    doc = results['documents'][0][i]

                    # parse memory data from document
                    import ast
                    memory_data = ast.literal_eval(doc)

                    memories.append({
                        "id": mem_id,
                        "score": score,
                        "memory_type": meta["memory_type"],
                        "memory_data": memory_data,
                        "text": meta["text"],
                        "timestamp": meta.get("timestamp")
                    })

        return memories

    def count(self) -> int:
        """total memories stored"""
        try:
            return self.collection.count()
        except:
            return 0

    def delete(self, memory_id: str) -> bool:
        """delete a memory by id"""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except:
            return False


# keep old name for backwards compatibility
QdrantStore = MemoryStore
