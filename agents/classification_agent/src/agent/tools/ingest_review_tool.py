"""Review Ingestion Tool - Parse and store raw reviews from other agents"""

from langchain_core.tools import tool
from agents.classification_agent.src.database import insert_new_review, get_next_review_id
from agents.classification_agent.src.utils import get_llm_from_config
import json
from datetime import datetime


@tool
def ingest_review(review_text: str) -> str:
    """
    Ingest a raw review from another agent and intelligently parse it for storage.

    This tool accepts raw review text (e.g., from a web scraper) and uses LLM to:
    - Extract reviewer name, rating, date, and other metadata from the text
    - Fill in missing fields with sensible defaults
    - Generate a unique review ID
    - Store the review in the database as unprocessed for later classification

    Args:
        review_text: The raw review text to parse and store

    Returns:
        JSON string with success status, review_id, and parsed fields
    """
    try:
        # Use LLM to extract structured fields from raw text
        llm = get_llm_from_config(temperature=0.0)

        extraction_prompt = f"""Extract the following fields from this review text. If a field is not present, return null.

Review text:
{review_text}

Extract and return ONLY a JSON object with these exact fields:
{{
  "reviewer_name": "full name if mentioned (e.g., 'John Doe', 'Sarah J.'), or null",
  "username": "username/handle if mentioned (e.g., '@johndoe'), or null",
  "email": "email if mentioned, or null",
  "rating": "numeric rating 1-5 (look for stars, '5/5', '⭐⭐⭐⭐⭐'), or null",
  "date": "date in YYYY-MM-DD format if mentioned, or null",
  "review_content": "the actual review text content, cleaned of metadata"
}}

Important:
- For rating: Convert star symbols (⭐), text ("5 stars"), or fractions ("5/5") to numbers 1-5
- For date: Convert relative dates ("2 days ago") to actual dates, or use date patterns
- Return ONLY the JSON object, no other text"""

        response = llm.invoke(extraction_prompt)
        parsed_data = json.loads(response.content)

        # Apply defaults for missing fields
        reviewer_name = parsed_data.get("reviewer_name") or "Anonymous"
        username = parsed_data.get("username") or "web_agent_user"
        email = parsed_data.get("email") or "scraped@agent.system"
        rating = parsed_data.get("rating")

        # Validate and default rating
        if rating is None or not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            rating = 3  # neutral default
        else:
            rating = int(rating)

        # Handle date
        date_str = parsed_data.get("date")
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Use cleaned review content if extracted, otherwise use original
        review_content = parsed_data.get("review_content") or review_text

        # Insert into database
        review_id = insert_new_review(
            review_text=review_content,
            username=username,
            email=email,
            reviewer_name=reviewer_name,
            rating=rating
        )

        result = {
            "success": True,
            "review_id": review_id,
            "parsed_fields": {
                "reviewer_name": reviewer_name,
                "username": username,
                "email": email,
                "rating": rating,
                "date": date_str,
                "review_length": len(review_content)
            },
            "message": f"Review {review_id} stored successfully and ready for classification"
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        # LLM didn't return valid JSON, fall back to simple insertion
        review_id = insert_new_review(
            review_text=review_text,
            username="web_agent_user",
            email="scraped@agent.system",
            reviewer_name="Anonymous",
            rating=3
        )

        return json.dumps({
            "success": True,
            "review_id": review_id,
            "parsed_fields": {"note": "Used default values due to parsing error"},
            "message": f"Review {review_id} stored with default values"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "Failed to ingest review"
        }, indent=2)
