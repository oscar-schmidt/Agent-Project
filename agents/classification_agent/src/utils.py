from pydantic import BaseModel, Field
from typing import List, Optional
import hashlib

# removed hardcoded criticality type to accept any string from LLM

#columns expteceted from the datasset as inputs
class RawReview(BaseModel):
    review_id: str
    review: str
    username: str
    email: str
    date: str
    reviewer_name: str
    rating: int

#detetcted error LLM output before it is normalised
# error_type is now any list of strings LLM-generated categories
# severity must be one of: Critical, Major, Minor, Suggestion, None
class DetectedError(BaseModel):
    error_summary: str
    error_type: List[str] = Field(default_factory=list) 
    severity: str = "None" 
    rationale: str

# Sentiment analysis data
class SentimentData(BaseModel):
    review_id: str
    overall_sentiment: str  # positive, negative or neutral
    overall_confidence: float  # 0.0-1.0
    sentiment_polarity: float  # -1.0 to +1.0

# normalised error - ready to log
class EnrichedError(BaseModel):
    review: RawReview
    error: DetectedError
    criticality: str  # Changed from Criticality Literal to str 
    error_hash: str

    #Sentiment enrichment
    sentiment_data: Optional[SentimentData] = None
    sentiment_influenced_criticality: bool = False

#deduplication
def hash_error(review_id: str, summary: str) -> str:
    key = f"{review_id}|{summary}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def get_llm_from_config(temperature: float = 0.0):
    """Create LLM instance from config.yaml"""
    from config.config_helper import get_model_config
    from langchain.chat_models import init_chat_model  # type: ignore

    config = get_model_config()
    return init_chat_model(f"{config['provider']}:{config['model']}", temperature=temperature)
