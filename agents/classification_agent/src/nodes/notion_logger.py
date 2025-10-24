# src/nodes/notion_logger.py
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from notion_client import Client
from notion_client.errors import APIResponseError



from agents.classification_agent.src.utils import EnrichedError

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID") or os.getenv("NOTION_DB_ID")

assert NOTION_API_KEY, "Missing NOTION_API_KEY"
assert NOTION_DATABASE_ID, "Missing NOTION_DATABASE_ID"

notion = Client(auth=NOTION_API_KEY)


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


#match properties from enrichederror to databses columns
def _props_from_enriched(e: EnrichedError, hash_value: str) -> Dict[str, Any]:
    review = e.review
    err = e.error

    # Base properties (always included)
    # Use review date if available, otherwise use current date when logging
    date_value = getattr(review, "date", None)
    if not date_value or not str(date_value).strip():
        # Use current date in ISO format (YYYY-MM-DD)
        date_value = datetime.now().strftime("%Y-%m-%d")

   
    categories_text = ", ".join(err.error_type)  # Join array into comma separated string

    props = {
        "ReviewID":     {"title":     [{"text": {"content": review.review_id}}]},
        "Reviewer":     {"rich_text": [{"text": {"content": getattr(review, "reviewer_name", "")}}]},
        "Date":         {"date":      {"start": str(date_value)}},
        "ReviewText":   {"rich_text": [{"text": {"content": review.review[:1900]}}]},
        "ErrorSummary": {"rich_text": [{"text": {"content": err.error_summary}}]},
        "Categories":   {"rich_text": [{"text": {"content": categories_text}}]},  
        "Severity":     {"select": {"name": e.criticality}},                      
        "Rationale":    {"rich_text": [{"text": {"content": err.rationale[:1900]}}]},
        "Hash":         {"rich_text": [{"text": {"content": hash_value}}]},
    }

    # Add sentiment fields if sentiment data exists
    if e.sentiment_data:
        try:
            sentiment_props = {
                # Changed from select to text for flexibility
                "OverallSentiment": {
                    "rich_text": [{"text": {"content": e.sentiment_data.overall_sentiment}}]
                },
                "SentimentScore": {
                    "number": round(e.sentiment_data.overall_confidence, 3)
                },
                "SentimentPolarity": {
                    "number": round(e.sentiment_data.sentiment_polarity, 3)
                },
                "SentimentInfluenced": {
                    "checkbox": e.sentiment_influenced_criticality
                },
                "Rating": {
                    "number": review.rating
                },
            }
            props.update(sentiment_props)
        except Exception as ex:
            # Gracefully skip if Notion DB doesn't have these fields yet
            print(f"Warning: Could not add sentiment fields to Notion: {ex}")
            print(f"Make sure your Notion database has the sentiment properties configured")

    return props


def _find_page_by_hash(hash_value: str) -> Optional[str]:
    res = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        filter={"property": "Hash", "rich_text": {"equals": hash_value}},
        page_size=1,
    )
    results = res.get("results", [])
    return results[0]["id"] if results else None

def upsert_enriched_error(e: EnrichedError) -> str:
    #use existing errorhash or compute one 
    hash_value = getattr(e, "error_hash", None) or _sha12(
        f"{e.review.review_id}|{e.error.error_summary}"
    )

    props = _props_from_enriched(e, hash_value)
    page_id = _find_page_by_hash(hash_value)

    # Create or update
    if page_id:
        page = notion.pages.update(page_id=page_id, properties=props)
    else:
        page = notion.pages.create(parent={"database_id": NOTION_DATABASE_ID}, properties=props)
    return page["id"]
