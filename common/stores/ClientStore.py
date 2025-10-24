import logging
from typing import Any, Literal, List
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from common.stores.Embedding import Embedding
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
import uuid



class ClientStore():
    def __init__(self, collection_name = "AgentRegistery") -> None:
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        self.embedding = Embedding()

    def _check_for_duplicates(self,vector: List[float]) -> str | None | int:
       
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

    def store(self, data: dict):
            vector = self.embedding.get_embedding(quary=data["description"])
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
            check = self._check_for_duplicates(vector=vector)
            if check is None:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload=data,
                        )
                    ]
                )
                return "Agent registration saved"
            else:
                return "Agent Registration already exists"
            
    def get(self, quary: dict) -> dict | None:
        vector = self.embedding.get_embedding(quary["description"])
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            limit=1, 
            query=vector,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.9,
        ).model_dump()

        agent_info = search_result
        return agent_info

    def update(self, agent_id: str, status: str) -> str:

        try:
            update_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="agent_id",
                        match=models.MatchValue(value=agent_id)
                    )
                ]
            )

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "status": status
                },
                points=update_filter,
                wait=True,
            )
            return f"Update successful"
        except Exception as e:
            return f"Update failed: {e}"










