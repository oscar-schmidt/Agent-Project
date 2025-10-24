from typing import List, Dict, Tuple
from agents.classification_agent.src.utils import RawReview, DetectedError, EnrichedError, SentimentData, hash_error

# removed import classify_criticality - now using llm generated severity directly


def apply_sentiment_adjustment(
    base_criticality: str,  
    sentiment: SentimentData,
    review_rating: int
) -> Tuple[str, bool]:  
    """
    Adjust criticality based on sentiment analysis and review rating.

    Args:
        base_criticality: The base criticality from LLM classification
        sentiment: Sentiment analysis data for the review
        review_rating: Customer rating (1-5)

    Returns:
        Tuple of (final_criticality, was_influenced)
    """
    polarity = sentiment.sentiment_polarity
    influenced = False
    final = base_criticality

    # Strong negative sentiment escalation
    # Severity values: Critical, Major, Minor, Suggestion, None
    if polarity < -0.85 and base_criticality == "Major":
        final = "Critical"
        influenced = True
    elif polarity < -0.70 and base_criticality == "Minor":
        final = "Major"
        influenced = True

    # Low rating escalation (1-2 stars with Major issues → Critical)
    if review_rating <= 2 and base_criticality == "Major":
        final = "Critical"
        influenced = True

    # Strong positive sentiment downgrade (positive review with minor issue)
    if polarity > 0.85 and base_criticality == "Minor":
        final = "Suggestion"
        influenced = True

    return final, influenced


#enrich error items with sentiment-aware criticality
def normalize(review: RawReview, detected: List[DetectedError], sentiment: SentimentData) -> List[EnrichedError]:
    """
    Normalize errors with sentiment-aware criticality classification.

    Args:
        review: The original review
        detected: List of detected errors (includes LLM generated severity)
        sentiment: Sentiment analysis data

    Returns:
        List of enriched errors with deduplicated hashes
    """
    enriched: List[EnrichedError] = []

    for e in detected:
        # CHANGED: Use LLM-generated severity from DetectedError instead of classify_criticality
        base_crit = e.severity  # LLM already assigned severity

        # Sentiment-adjusted classification (optional escalation/downgrade)
        final_crit, influenced = apply_sentiment_adjustment(
            base_crit,
            sentiment,
            review.rating
        )

        enriched.append(
            EnrichedError(
                review=review,
                error=e,
                criticality=final_crit,
                error_hash=hash_error(review.review_id, e.error_summary),
                sentiment_data=sentiment,
                sentiment_influenced_criticality=influenced
            )
        )

    # Deduplicate by hash
    uniq: Dict[str, EnrichedError] = {}
    for it in enriched:
        uniq[it.error_hash] = it
    return list(uniq.values())
