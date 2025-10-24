"""
Log to Notion Database Tool
Tool to log processed reviews with sentiment scores and severity classification to Notion database.
"""

import os
import json
from typing import List, Optional
from langchain_core.tools import BaseTool

from agents.classification_agent.src.utils import RawReview, DetectedError, EnrichedError, SentimentData, hash_error
from agents.classification_agent.src.nodes.notion_logger import upsert_enriched_error
from agents.classification_agent.src.database import mark_reviews_processed


class NotionTool(BaseTool):
    """Tool for logging review analysis results to Notion database"""

    name: str = "log_reviews_to_notion"
    description: str = "Log processed reviews to Notion database for team tracking and collaboration."

    # Pydantic fields - must be declared as class attributes
    api_key: Optional[str] = None
    database_id: Optional[str] = None
    use_database: bool = False
    global_dry_run: bool = False

    def __init__(
        self,
        api_key: str = None,
        database_id: str = None,
        use_database: bool = None,
        global_dry_run: bool = None,
        **kwargs
    ):
        """
        Initialize the NotionTool

        Args:
            api_key: Notion API key (defaults to env var)
            database_id: Notion database ID (defaults to env var)
            use_database: Whether database marking is enabled (defaults to env var)
            global_dry_run: Dry run mode (defaults to env var)
        """
        # Set defaults before calling super().__init__()
        if api_key is None:
            api_key = os.getenv("NOTION_API_KEY")

        if database_id is None:
            database_id = os.getenv("NOTION_DATABASE_ID")

        if use_database is None:
            use_database = os.getenv("USE_DATABASE", "false").lower() == "true"

        if global_dry_run is None:
            global_dry_run = os.getenv("NOTION_DRY_RUN", "0") in ("1", "true", "True")

        super().__init__(
            api_key=api_key,
            database_id=database_id,
            use_database=use_database,
            global_dry_run=global_dry_run,
            **kwargs
        )

    def _parse_review_data(self, review_data_json: str) -> List[EnrichedError]:
        """
        Parse review data JSON and convert to EnrichedError objects

        Args:
            review_data_json: JSON string containing review and sentiment data

        Returns:
            List of EnrichedError objects

        Raises:
            ValueError: If JSON is invalid
        """
        try:
            data = json.loads(review_data_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        enriched_errors = []

        reviews_list = data.get("reviews", [])
        sentiments_list = data.get("sentiments", [])

        # Build sentiment map by review ID
        sentiment_map = {}
        for sent_item in sentiments_list:
            review_id = sent_item.get("review_id")
            if review_id and "sentiment" in sent_item:
                s = sent_item["sentiment"]
                sentiment_map[review_id] = SentimentData(
                    review_id=review_id,
                    overall_sentiment=s.get("overall_sentiment", "Neutral"),
                    overall_confidence=s.get("confidence", 0.0),
                    sentiment_polarity=s.get("polarity", 0.0)
                )

        # Process each review
        for review_item in reviews_list:
            review_id = review_item.get("review_id")
            if not review_id:
                continue

            review = RawReview(
                review_id=review_id,
                review=review_item.get("review_text", ""),
                username=review_item.get("username", ""),
                email=review_item.get("email", ""),
                date=review_item.get("date", ""),
                reviewer_name=review_item.get("reviewer_name", ""),
                rating=review_item.get("rating", 3)
            )

            # Get sentiment data for this review
            sentiment_data = sentiment_map.get(review_id)

            # Process each error in the review
            errors = review_item.get("errors", [])
            for error_item in errors:
                # Handle case where error_item might be a string instead of dict
                if isinstance(error_item, str):
                    # Skip string errors - they're not properly formatted
                    continue

                # Support both old and new field names
                categories = error_item.get("categories") or error_item.get("error_type", ["Other"])
                severity = error_item.get("severity") or error_item.get("criticality", "Medium Priority")

                detected_error = DetectedError(
                    error_summary=error_item.get("error_summary", ""),
                    error_type=categories,  # Maps to categories
                    severity=severity,       # LLM-generated severity
                    rationale=error_item.get("rationale", "")
                )

                criticality = severity  # Use severity as criticality for EnrichedError

                enriched_errors.append(
                    EnrichedError(
                        review=review,
                        error=detected_error,
                        criticality=criticality,
                        error_hash=hash_error(review_id, detected_error.error_summary),
                        sentiment_data=sentiment_data,
                        sentiment_influenced_criticality=False  # Can be enhanced later
                    )
                )

        return enriched_errors

    def _log_to_notion(
        self,
        enriched_errors: List[EnrichedError],
        dry_run: bool
    ) -> tuple:
        """
        Log enriched errors to Notion

        Args:
            enriched_errors: List of EnrichedError objects to log
            dry_run: If True, only simulate logging

        Returns:
            Tuple of (logged_count, processed_review_ids)
        """
        processed_review_ids = []
        logged_count = 0

        for enriched_error in enriched_errors:
            if dry_run:
                print(f"[DRY RUN] Would upsert: {enriched_error.review.review_id} | "
                      f"{enriched_error.error.error_summary} | {enriched_error.criticality}")
            else:
                # Actually write to Notion
                try:
                    print(f"  Logging {enriched_error.review.review_id} to Notion...")
                    upsert_enriched_error(enriched_error)
                    print(f"  Successfully logged {enriched_error.review.review_id}")
                except Exception as notion_error:
                    print(f"  ERROR logging {enriched_error.review.review_id}: "
                          f"{type(notion_error).__name__}: {str(notion_error)}")
                    import traceback
                    traceback.print_exc()
                    raise

            logged_count += 1
            if enriched_error.review.review_id not in processed_review_ids:
                processed_review_ids.append(enriched_error.review.review_id)

        return logged_count, processed_review_ids

    def _run(
        self,
        review_data: str,
        include_sentiment: bool = True,
        include_criticality: bool = True,
        dry_run: bool = False
    ) -> str:
        """
        Run the Notion logging tool

        Args:
            review_data: JSON string containing review data with classifications/sentiments
            include_sentiment: Whether to include sentiment data (unused, kept for compatibility)
            include_criticality: Whether to include criticality data (unused, kept for compatibility)
            dry_run: If True, simulate logging without actually writing to Notion

        Returns:
            JSON string with logging results
        """
        try:
            # Check if Notion is configured
            if not self.api_key or not self.database_id:
                return json.dumps({
                    "success": False,
                    "error": "Notion is not configured. Set NOTION_API_KEY and NOTION_DATABASE_ID in .env file",
                    "total_logged": 0
                })

            # Determine dry run mode (parameter or global setting)
            dry_run = dry_run or self.global_dry_run

            # Parse the review data
            enriched_errors = self._parse_review_data(review_data)

            if not enriched_errors:
                return json.dumps({
                    "success": False,
                    "message": "No valid review data found to log",
                    "total_logged": 0
                })

            # Log to Notion
            logged_count, processed_review_ids = self._log_to_notion(enriched_errors, dry_run)

            # Mark reviews as processed in database
            if self.use_database and processed_review_ids and not dry_run:
                mark_reviews_processed(processed_review_ids)

            return json.dumps({
                "success": True,
                "total_logged": logged_count,
                "total_reviews": len(processed_review_ids),
                "review_ids": processed_review_ids,
                "dry_run": dry_run,
                "message": f"{'[DRY RUN] Would log' if dry_run else 'Successfully logged'} "
                          f"{logged_count} error(s) from {len(processed_review_ids)} review(s) to Notion"
            }, indent=2)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in log_reviews_to_notion: {type(e).__name__}: {str(e)}")
            print(error_details)
            return json.dumps({
                "success": False,
                "error": f"Failed to log reviews to Notion: {type(e).__name__}: {str(e)}",
                "error_type": type(e).__name__,
                "total_logged": 0
            })


# Create default instance for backward compatibility
log_reviews_to_notion = NotionTool()
