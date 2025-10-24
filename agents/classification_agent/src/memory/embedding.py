from sentence_transformers import SentenceTransformer
from typing import Union, List
from .schemas import Episode, Semantic


class EmbeddingGenerator:
    """generates embeddings using sentence transformers - runs locally, no api needed"""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        # lightweight model, 384 dimensions, good for semantic search
        # downloads automatically on first use
        self.model = SentenceTransformer(model)

    def memory_to_text(self, memory: Union[Episode, Semantic]) -> str:
        """convert memory obj to text for embedding"""
        if isinstance(memory, Episode):
            return f"{memory.observation} {memory.thoughts} {memory.action} {memory.result}"
        elif isinstance(memory, Semantic):
            ctx = f" {memory.context}" if memory.context else ""
            return f"{memory.subject} {memory.predicate} {memory.object}{ctx}"
        return str(memory)

    def generate(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """generate embedding vector"""
        is_single = isinstance(text, str)

        if is_single:
            # for single string, encode directly returns 1d array
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # for list, encode returns 2d array
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]

    def generate_query_embedding(self, query: str) -> List[float]:
        """generate embedding for search queries - same as document for this model"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
