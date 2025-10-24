#!/usr/bin/env python3
"""
Test script for sentiment analysis integration.

Tests the sentiment analysis functionality with sample reviews
to verify the implementation is working correctly.
"""

import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variables for testing
os.environ["ENABLE_SENTIMENT"] = "true"
os.environ["NOTION_DRY_RUN"] = "1"  # Dry run mode to avoid writing to Notion

from src.utils import RawReview, SentimentData
from src.nodes.sentiment_analysis import analyze_review_sentiment
from src.nodes.normalize import apply_sentiment_adjustment


def test_sentiment_analysis():
    """Test sentiment analysis with sample reviews"""
    print("="*60)
    print("Testing Sentiment Analysis Integration")
    print("="*60)
    print()

    # Test Case 1: Negative review with crash
    print("Test 1: Negative Review with Crash")
    print("-" * 60)
    review1 = RawReview(
        review_id="TEST-001",
        review="The app keeps crashing when I switch workspaces. I've lost my work twice! Very frustrating and disappointing.",
        username="test_user1",
        email="test1@example.com",
        date="2024-10-10",
        reviewer_name="Test User 1",
        rating=1
    )

    try:
        sentiment1 = analyze_review_sentiment(review1)
        print(f"Review: {review1.review[:80]}...")
        print(f"Rating: {review1.rating} stars")
        print(f"Sentiment: {sentiment1.overall_sentiment}")
        print(f"Confidence: {sentiment1.overall_confidence:.3f}")
        print(f"Polarity: {sentiment1.sentiment_polarity:.3f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Test Case 2: Positive review
    print("Test 2: Positive Review")
    print("-" * 60)
    review2 = RawReview(
        review_id="TEST-002",
        review="Love the new UI! It's so clean and intuitive. Great job on the update!",
        username="test_user2",
        email="test2@example.com",
        date="2024-10-10",
        reviewer_name="Test User 2",
        rating=5
    )

    try:
        sentiment2 = analyze_review_sentiment(review2)
        print(f"Review: {review2.review[:80]}...")
        print(f"Rating: {review2.rating} stars")
        print(f"Sentiment: {sentiment2.overall_sentiment}")
        print(f"Confidence: {sentiment2.overall_confidence:.3f}")
        print(f"Polarity: {sentiment2.sentiment_polarity:.3f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Test Case 3: Mixed review
    print("Test 3: Mixed Review")
    print("-" * 60)
    review3 = RawReview(
        review_id="TEST-003",
        review="The features are great but the app is sometimes slow. Overall decent experience.",
        username="test_user3",
        email="test3@example.com",
        date="2024-10-10",
        reviewer_name="Test User 3",
        rating=3
    )

    try:
        sentiment3 = analyze_review_sentiment(review3)
        print(f"Review: {review3.review[:80]}...")
        print(f"Rating: {review3.rating} stars")
        print(f"Sentiment: {sentiment3.overall_sentiment}")
        print(f"Confidence: {sentiment3.overall_confidence:.3f}")
        print(f"Polarity: {sentiment3.sentiment_polarity:.3f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()


def test_sentiment_adjustment():
    """Test sentiment-based criticality adjustment"""
    print("="*60)
    print("Testing Criticality Adjustment Logic")
    print("="*60)
    print()

    # Test Case 1: Strong negative sentiment escalates Major to Critical
    print("Test 1: Strong Negative Escalation")
    print("-" * 60)
    sentiment_negative = SentimentData(
        review_id="TEST-001",
        overall_sentiment="Negative",
        overall_confidence=0.95,
        sentiment_polarity=-0.95
    )
    final, influenced = apply_sentiment_adjustment("Major", sentiment_negative, 1)
    print(f"Base: Major, Sentiment: {sentiment_negative.sentiment_polarity:.2f}, Rating: 1")
    print(f"Final: {final}, Influenced: {influenced}")
    print(f"Expected: Critical, Influenced: True")
    print(f"✓ PASS" if final == "Critical" and influenced else "✗ FAIL")
    print()

    # Test Case 2: Low rating escalates Major to Critical
    print("Test 2: Low Rating Escalation")
    print("-" * 60)
    sentiment_neutral = SentimentData(
        review_id="TEST-002",
        overall_sentiment="Neutral",
        overall_confidence=0.6,
        sentiment_polarity=0.0
    )
    final, influenced = apply_sentiment_adjustment("Major", sentiment_neutral, 2)
    print(f"Base: Major, Sentiment: {sentiment_neutral.sentiment_polarity:.2f}, Rating: 2")
    print(f"Final: {final}, Influenced: {influenced}")
    print(f"Expected: Critical, Influenced: True")
    print(f"✓ PASS" if final == "Critical" and influenced else "✗ FAIL")
    print()

    # Test Case 3: Strong positive downgrades Minor to Suggestion
    print("Test 3: Positive Downgrade")
    print("-" * 60)
    sentiment_positive = SentimentData(
        review_id="TEST-003",
        overall_sentiment="Positive",
        overall_confidence=0.90,
        sentiment_polarity=0.90
    )
    final, influenced = apply_sentiment_adjustment("Minor", sentiment_positive, 5)
    print(f"Base: Minor, Sentiment: {sentiment_positive.sentiment_polarity:.2f}, Rating: 5")
    print(f"Final: {final}, Influenced: {influenced}")
    print(f"Expected: Suggestion, Influenced: True")
    print(f"✓ PASS" if final == "Suggestion" and influenced else "✗ FAIL")
    print()

    # Test Case 4: No adjustment needed
    print("Test 4: No Adjustment")
    print("-" * 60)
    sentiment_mild = SentimentData(
        review_id="TEST-004",
        overall_sentiment="Neutral",
        overall_confidence=0.7,
        sentiment_polarity=-0.3
    )
    final, influenced = apply_sentiment_adjustment("Minor", sentiment_mild, 3)
    print(f"Base: Minor, Sentiment: {sentiment_mild.sentiment_polarity:.2f}, Rating: 3")
    print(f"Final: {final}, Influenced: {influenced}")
    print(f"Expected: Minor, Influenced: False")
    print(f"✓ PASS" if final == "Minor" and not influenced else "✗ FAIL")
    print()


if __name__ == "__main__":
    print("\n")
    print("🔬 Sentiment Analysis Test Suite")
    print("=" * 60)
    print()

    try:
        # Test sentiment analysis
        test_sentiment_analysis()

        # Test criticality adjustment
        test_sentiment_adjustment()

        print("="*60)
        print("✅ All tests completed!")
        print("="*60)
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full pipeline with: python src/run.py")
        print("3. Use NOTION_DRY_RUN=1 to preview results without writing")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
