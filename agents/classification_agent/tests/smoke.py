# tests/smoke.py
from src.config import OLLAMA_MODEL
from src.database import load_unprocessed_reviews
from src.nodes.detect_errors import detect_errors_with_ollama
from src.nodes.normalize import normalize
from src.utils import SentimentData

if __name__ == "__main__":
    reviews = load_unprocessed_reviews(batch_size=3)
    print(f"Loaded {len(reviews)} unprocessed reviews from database")

    for i, r in enumerate(reviews, start=1):
        errs = detect_errors_with_ollama(r, OLLAMA_MODEL)
        # Create dummy sentiment data for testing
        sentiment = SentimentData(
            review_id=r.review_id,
            overall_sentiment="Neutral",
            overall_confidence=0.0,
            sentiment_polarity=0.0
        )
        enriched = normalize(r, errs, sentiment)

        print(f"\nReview #{i} | id={r.review_id} | rating={r.rating}")
        print(r.review[:220] + ("..." if len(r.review) > 220 else ""))

        if not enriched:
            print("→ No errors detected.")
        else:
            for e in enriched:
                print(f"→ [{e.criticality}] {e.error.error_summary} | types={e.error.error_type} | hash={e.error_hash}")
