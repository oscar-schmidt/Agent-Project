"""
Criticality Classification Tool
Classify customer reviews by detecting errors and assigning severity levels.
"""

import json
from typing import Optional, List, Any, Dict
from langchain_core.tools import BaseTool

from agents.classification_agent.src.config import AGENT_MODEL
from agents.classification_agent.src.utils import RawReview
from agents.classification_agent.src.nodes.detect_errors import detect_errors_with_ollama
from agents.classification_agent.src.database import load_unprocessed_reviews, get_connection


class CriticalityTool(BaseTool):
    """Tool for classifying review criticality and detecting errors"""

    name: str = "classify_review_criticality"
    description: str = "Classify criticality of customer reviews and detect errors/issues using LLM analysis."

    # Pydantic fields - must be declared as class attributes
    llm_model: str = AGENT_MODEL  # use same LLM model as agent
    batch_size: int = 50
    _cache: Dict[str, Any] = {}

    def __init__(
        self,
        llm_model: str = None,
        batch_size: int = 50,
        **kwargs
    ):
        """
        Initialize the CriticalityTool

        Args:
            llm_model: LLM model to use for classification (defaults to config)
            batch_size: Maximum batch size for processing
        """
        # Set defaults before calling super().__init__()
        if llm_model is None:
            llm_model = AGENT_MODEL

        super().__init__(
            llm_model=llm_model,
            batch_size=batch_size,
            _cache={},
            **kwargs
        )

    def _load_reviews_by_ids(self, review_ids: List[str]) -> List[RawReview]:
        """
        Load specific reviews by their IDs from the database.

        Args:
            review_ids: List of review IDs to load

        Returns:
            List of RawReview objects
        """
        if not review_ids:
            return []

        conn = get_connection()
        cursor = conn.cursor()

        placeholders = ','.join(['%s'] * len(review_ids))
        query = f"""
            SELECT review_id, review, username, email, date, reviewer_name, rating
            FROM raw_reviews
            WHERE review_id IN ({placeholders})
            ORDER BY created_at DESC
        """

        cursor.execute(query, review_ids)
        rows = cursor.fetchall()

        reviews = []
        for row in rows:
            # Convert datetime to string
            date_str = row[4]
            if hasattr(date_str, 'strftime'):
                date_str = date_str.strftime('%Y-%m-%d %H:%M:%S')

            reviews.append(RawReview(
                review_id=row[0],
                review=row[1],
                username=row[2],
                email=row[3],
                date=str(date_str),
                reviewer_name=row[5],
                rating=row[6]
            ))

        cursor.close()
        conn.close()

        return reviews

    def _load_unprocessed_reviews(self, limit: int) -> List[RawReview]:
        """
        Load unprocessed reviews from database

        Args:
            limit: Maximum number of reviews to load

        Returns:
            List of RawReview objects
        """
        return load_unprocessed_reviews(batch_size=limit)

    def _classify_errors(self, reviews: List[RawReview]) -> List[dict]:
        """
        Classify errors for a list of reviews

        Args:
            reviews: List of reviews to process

        Returns:
            List of classification results
        """
        results = []

        for review in reviews:
            # Detect errors and severity using LLM
            detected_errors = detect_errors_with_ollama(review, self.llm_model)

            classified_errors = []
            for error in detected_errors:
                # LLM generated severity instead of classify_criticality
                classified_errors.append({
                    "error_summary": error.error_summary,
                    "categories": error.error_type,
                    "severity": error.severity,  # LLM generated severity
                    "rationale": error.rationale
                })

                # Save error to database
                from agents.classification_agent.src.database import insert_detected_error
                print(f"[DEBUG] Inserting error to DB: {review.review_id} - {error.error_summary}")
                success = insert_detected_error(
                    review_id=review.review_id,
                    error_summary=error.error_summary,
                    error_type=error.error_type,
                    criticality=error.severity,
                    rationale=error.rationale
                )
                if success:
                    print(f"[DEBUG] Successfully inserted error for {review.review_id}")
                else:
                    print(f"[DEBUG] Failed to insert error for {review.review_id}")

            results.append({
                "review_id": review.review_id,
                "review_text": review.review[:200] + "..." if len(review.review) > 200 else review.review,
                "rating": review.rating,
                "reviewer_name": review.reviewer_name,
                "errors": classified_errors,
                "error_count": len(classified_errors)
            })

        return results

    def _run(
        self,
        review_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> str:
        """
        Run the criticality classification tool

        Args:
            review_ids: Optional list of specific review IDs to process
            limit: Maximum number of reviews to process (default 10, max from batch_size)

        Returns:
            JSON string with classification results
        """
        try:
            # Cap limit for performance
            limit = min(limit, self.batch_size)

            # Load reviews
            if review_ids:
                reviews = self._load_reviews_by_ids(review_ids)
            else:
                reviews = self._load_unprocessed_reviews(limit)

            if not reviews:
                return json.dumps({
                    "message": "No reviews found to process",
                    "reviews": [],
                    "total_processed": 0
                })

            # Classify errors
            results = self._classify_errors(reviews)

            return json.dumps({
                "reviews": results,
                "total_processed": len(results),
                "review_ids": [r.review_id for r in reviews]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Failed to classify reviews: {str(e)}",
                "reviews": [],
                "total_processed": 0
            })


# Create default instance for backward compatibility
classify_review_criticality = CriticalityTool()
