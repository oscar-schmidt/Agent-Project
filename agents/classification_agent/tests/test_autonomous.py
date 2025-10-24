#!/usr/bin/env python3
"""
Test script for autonomous classification system.
Tests with diverse review types to verify LLM generates appropriate categories.
"""

import sys
import os
# Add parent directory to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import RawReview
from src.nodes.detect_errors import detect_errors_with_ollama

def test_review(review: RawReview, expected_business_type: str = None):
    """Test a single review and print results."""
    print(f"\n{'='*60}")
    print(f"Review: {review.review}")
    print(f"Rating: {review.rating}/5")
    print(f"{'='*60}")

    try:
        errors = detect_errors_with_ollama(review)

        if not errors:
            print("✅ No issues detected")
            return

        for i, e in enumerate(errors, 1):
            print(f"\nIssue #{i}:")
            print(f"  Categories: {', '.join(e.error_type)}")
            print(f"  Severity: {e.severity}")
            print(f"  Summary: {e.error_summary}")
            print(f"  Rationale: {e.rationale}")

    except Exception as ex:
        print(f"❌ Error: {ex}")

def main():
    """Run test suite with diverse business types."""

    print("🧪 Testing Autonomous Classification System")
    print("This will test the LLM's ability to adapt to different business types.\n")

    # Test 1: Restaurant Review
    print("\n" + "🍔 TEST 1: RESTAURANT REVIEW " + "="*40)
    test_review(RawReview(
        review_id="TEST-001",
        review="The BBQ tasting palette did not live up to the hype. Food was cold and service was slow. Waited 45 minutes for our table.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="John D.",
        rating=2
    ), expected_business_type="restaurant")

    # Test 2: Software/SaaS Review
    print("\n" + "💻 TEST 2: SOFTWARE REVIEW " + "="*40)
    test_review(RawReview(
        review_id="TEST-002",
        review="The system crashes very often when uploading large files. Also the UI is confusing and login times out frequently.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="Sarah K.",
        rating=1
    ), expected_business_type="software")

    # Test 3: Hotel Review
    print("\n" + "🏨 TEST 3: HOTEL REVIEW " + "="*40)
    test_review(RawReview(
        review_id="TEST-003",
        review="Room was dirty and the AC didn't work. When I complained to the front desk, they were rude and unhelpful.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="Mike T.",
        rating=1
    ), expected_business_type="hotel")

    # Test 4: E-commerce Review
    print("\n" + "📦 TEST 4: E-COMMERCE REVIEW " + "="*40)
    test_review(RawReview(
        review_id="TEST-004",
        review="Product arrived damaged and took 2 weeks longer than promised. Customer service was no help with the return.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="Lisa R.",
        rating=1
    ), expected_business_type="retail")

    # Test 5: Feature Request (Software)
    print("\n" + "💡 TEST 5: FEATURE REQUEST " + "="*40)
    test_review(RawReview(
        review_id="TEST-005",
        review="Love the app! Would be nice if you could add dark mode and bulk edit features.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="Alex P.",
        rating=5
    ), expected_business_type="software")

    # Test 6: Positive Review (No Issues)
    print("\n" + "⭐ TEST 6: POSITIVE REVIEW " + "="*40)
    test_review(RawReview(
        review_id="TEST-006",
        review="Amazing experience! The food was delicious, service was excellent, and ambiance was perfect.",
        username="test",
        email="test@example.com",
        date="2025-01-15",
        reviewer_name="Emma W.",
        rating=5
    ), expected_business_type="restaurant")

    print("\n" + "="*60)
    print("🎉 Test suite complete!")
    print("\nWhat to look for:")
    print("  ✅ Restaurant reviews should get food/service categories")
    print("  ✅ Software reviews should get stability/UI categories")
    print("  ✅ Hotel reviews should get room/staff categories")
    print("  ✅ E-commerce reviews should get shipping/product categories")
    print("  ✅ Feature requests should have 'Suggestion' or low severity")
    print("  ✅ Positive reviews should have no issues or very low severity")
    print("\n")

if __name__ == "__main__":
    main()
