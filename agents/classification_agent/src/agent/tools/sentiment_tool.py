"""
Sentiment Analysis Tool
Analyzes emotional tone and sentiment of customer reviews using DeBERTa transformer model.
"""

import os
import json
from typing import Optional, List, Any, Dict
from langchain_core.tools import BaseTool

from agents.classification_agent.src.utils import RawReview
from agents.classification_agent.src.nodes.sentiment_analysis import analyze_review_sentiment as analyze_sentiment_node
from agents.classification_agent.src.database import load_unprocessed_reviews


class SentimentTool(BaseTool):
    """Tool for analyzing review sentiment using DeBERTa transformer model"""

    name: str = "analyze_review_sentiment"
    description: str = "Analyze sentiment and emotional tone of customer reviews using DeBERTa transformer model."

    # Pydantic fields - must be declared as class attributes
    sentiment_enabled: bool = True
    use_database: bool = False
    batch_size: int = 50
    data_path: str = ""
    _cache: Dict[str, Any] = {}

    def __init__(
        self,
        sentiment_enabled: bool = None,
        use_database: bool = None,
        batch_size: int = 50,
        data_path: str = None,
        **kwargs
    ):
        """
        Initialize the SentimentTool

        Args:
            sentiment_enabled: Whether sentiment analysis is enabled (defaults to env var)
            use_database: Whether to use database or CSV (defaults to env var)
            batch_size: Maximum batch size for processing
            data_path: Path to CSV file if not using database
        """
        # Set defaults before calling super().__init__()
        if sentiment_enabled is None:
            sentiment_enabled = os.getenv("ENABLE_SENTIMENT", "true").lower() in ("true", "1", "yes")

        if use_database is None:
            use_database = os.getenv("USE_DATABASE", "false").lower() == "true"

        if data_path is None:
            from agents.classification_agent.src.config import DATA_PATH
            data_path = DATA_PATH

        super().__init__(
            sentiment_enabled=sentiment_enabled,
            use_database=use_database,
            batch_size=batch_size,
            data_path=data_path,
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

        from agents.classification_agent.src.database import get_connection

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
        Load unprocessed reviews from database or CSV

        Args:
            limit: Maximum number of reviews to load

        Returns:
            List of RawReview objects
        """
        if self.use_database:
            return load_unprocessed_reviews(batch_size=limit)
        else:
            # Fallback to CSV
            from agents.classification_agent.src.nodes.load_reviews import load_reviews
            all_reviews = load_reviews(self.data_path)
            return all_reviews[:limit]

    def _analyze_sentiments(self, reviews: List[RawReview]) -> List[dict]:
        """
        Analyze sentiment for a list of reviews

        Args:
            reviews: List of reviews to process

        Returns:
            List of sentiment analysis results
        """
        results = []

        for idx, review in enumerate(reviews, 1):
            try:
                sentiment_data = analyze_sentiment_node(review)

                results.append({
                    "review_id": review.review_id,
                    "review_text": review.review[:200] + "..." if len(review.review) > 200 else review.review,
                    "rating": review.rating,
                    "reviewer_name": review.reviewer_name,
                    "sentiment": {
                        "overall_sentiment": sentiment_data.overall_sentiment,
                        "confidence": round(sentiment_data.overall_confidence, 4),
                        "polarity": round(sentiment_data.sentiment_polarity, 4)
                    }
                })

                # Progress logging
                if idx % 10 == 0:
                    print(f"Analyzed {idx}/{len(reviews)} reviews")

            except Exception as inner_error:
                # Log the specific error for this review
                print(f"  ERROR analyzing review {review.review_id}: {type(inner_error).__name__}: {str(inner_error)}")
                import traceback
                traceback.print_exc()

                # Continue with other reviews but include this error in results
                results.append({
                    "review_id": review.review_id,
                    "review_text": review.review[:200] + "..." if len(review.review) > 200 else review.review,
                    "rating": review.rating,
                    "reviewer_name": review.reviewer_name,
                    "sentiment": {
                        "error": f"{type(inner_error).__name__}: {str(inner_error)}"
                    }
                })

        return results

    def _run(
        self,
        review_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> str:
        """
        Run the sentiment analysis tool

        Args:
            review_ids: Optional list of specific review IDs to process
            limit: Maximum number of reviews to process (default 10, max from batch_size)

        Returns:
            JSON string with sentiment analysis results
        """
        try:
            # Check if sentiment analysis is enabled
            if not self.sentiment_enabled:
                return json.dumps({
                    "error": "Sentiment analysis is disabled (ENABLE_SENTIMENT=false in config)",
                    "sentiments": [],
                    "total_analyzed": 0
                })

            # Cap limit for performance
            limit = min(limit, self.batch_size)

            # Load reviews
            if review_ids:
                if not self.use_database:
                    return json.dumps({
                        "error": "Cannot load specific review IDs without database enabled (USE_DATABASE=true)",
                        "sentiments": [],
                        "total_analyzed": 0
                    })
                reviews = self._load_reviews_by_ids(review_ids)
            else:
                reviews = self._load_unprocessed_reviews(limit)

            if not reviews:
                return json.dumps({
                    "message": "No reviews found to analyze",
                    "sentiments": [],
                    "total_analyzed": 0
                })

            # Analyze sentiments
            results = self._analyze_sentiments(reviews)

            # Calculate summary statistics
            successful_results = [r for r in results if "error" not in r.get("sentiment", {})]

            return json.dumps({
                "sentiments": results,
                "total_analyzed": len(results),
                "review_ids": [r.review_id for r in reviews],
                "summary": {
                    "positive": sum(1 for r in successful_results if r["sentiment"]["overall_sentiment"] == "Positive"),
                    "negative": sum(1 for r in successful_results if r["sentiment"]["overall_sentiment"] == "Negative"),
                    "neutral": sum(1 for r in successful_results if r["sentiment"]["overall_sentiment"] == "Neutral"),
                    "avg_polarity": round(sum(r["sentiment"]["polarity"] for r in successful_results) / len(successful_results), 4) if successful_results else 0
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze sentiment: {str(e)}",
                "sentiments": [],
                "total_analyzed": 0
            })


# Create default instance for backward compatibility
analyze_review_sentiment = SentimentTool()
