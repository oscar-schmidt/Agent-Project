from pydantic import BaseModel

class Semantic(BaseModel):
    """stores all new facts, prefernecs and relationshops as triples"""
    subject: str
    predicate: str
    object: str
    context: str | None = None
