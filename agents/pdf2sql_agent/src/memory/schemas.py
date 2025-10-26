from pydantic import BaseModel, Field
from typing import Optional


class Episode(BaseModel):
    """stores agent experiences from classification tasks"""

    observation: str = Field(description="what happend - user request and review content")
    thoughts: str = Field(description="why we chose this classification")
    action: str = Field(description="what tools we called")
    result: str = Field(description="outcome and any feedback from user")


class Semantic(BaseModel):
    """stores facts as triples - subject predicate object"""

    subject: str
    predicate: str
    object: str
    context: Optional[str] = None  # when does this apply
