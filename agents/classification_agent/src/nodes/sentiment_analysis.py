"""
Sentiment Analysis Node for Classification Agent

Provides sentiment analysis functionality integrated with the review classification pipeline.
Uses DeBERTa model for aspect-based sentiment analysis.
"""

import os
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from agents.classification_agent.src.utils import RawReview, SentimentData


class SentimentAnalyzer:
    """
    Singleton sentiment analyzer to avoid reloading model multiple times.

    This class loads the DeBERTa model once and reuses it for all analyses.
    First initialization will download the model (~1-2 minutes).
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, confidence_threshold=0.8, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        """
        Initialize the sentiment analyzer (only runs once due to singleton pattern).

        Args:
            confidence_threshold (float): Minimum confidence for results (default: 0.8)
            model_name (str): HuggingFace model name for sentiment analysis
        """
        # Only initialize once
        if SentimentAnalyzer._initialized:
            return

        print("Loading sentiment model (first run only)")
        print(f"   Model: {model_name}")
        print("   (This may take 1-2 minutes on first run)")

        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        # Load tokenizer and model (use slow tokenizer to avoid conversion issues)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )

        print("Sentiment model loaded successfully")
        SentimentAnalyzer._initialized = True

    def truncate_text(self, text: str, max_tokens: int = 450) -> str:
        """
        Intelligently truncate text to fit within token limits.
        Keeps first 200 + last 250 tokens to preserve context.

        Args:
            text (str): Text to truncate
            max_tokens (int): Maximum tokens to keep (default: 450 to stay under 512 limit)

        Returns:
            str: Truncated text
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            return text

        # Keep first 200 and last 250 tokens
        start_tokens = tokens[:200]
        end_tokens = tokens[-250:]

        start_text = self.tokenizer.decode(start_tokens, skip_special_tokens=True)
        end_text = self.tokenizer.decode(end_tokens, skip_special_tokens=True)

        return f"{start_text} [...] {end_text}"

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Sentiment result with 'label' and 'score'
                  Example: {"label": "Positive", "score": 0.95}
        """
        # Truncate if necessary
        truncated_text = self.truncate_text(text)

        # Analyze sentiment
        result = self.sentiment_pipeline(truncated_text)[0]
        return result


def analyze_review_sentiment(review: RawReview) -> SentimentData:
    """
    Analyze sentiment for a single review.

    Args:
        review (RawReview): The review to analyze

    Returns:
        SentimentData: Sentiment analysis results
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(review.review)

    label = result['label']
    score = result['score']

    # Calculate polarity (-1 to +1)
    # Positive sentiment → positive polarity
    # Negative sentiment → negative polarity
    # Neutral → 0
    if label == "Positive":
        polarity = score
    elif label == "Negative":
        polarity = -score
    else:
        polarity = 0.0

    return SentimentData(
        review_id=review.review_id,
        overall_sentiment=label,
        overall_confidence=score,
        sentiment_polarity=polarity
    )


def analyze_batch_sentiments(reviews: List[RawReview]) -> List[SentimentData]:
    """
    Analyze sentiment for a batch of reviews.

    Args:
        reviews (List[RawReview]): List of reviews to analyze

    Returns:
        List[SentimentData]: List of sentiment analysis results
    """
    print(f" Analyzing sentiment for {len(reviews)} reviews...")

    results = []
    for idx, review in enumerate(reviews, 1):
        sentiment = analyze_review_sentiment(review)
        results.append(sentiment)

        if idx % 10 == 0:
            print(f"  ... {idx}/{len(reviews)} analyzed")

    print("Sentiment analysis complete")
    return results
