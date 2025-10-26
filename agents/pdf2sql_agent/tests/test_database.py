#!/usr/bin/env python3
"""
Simple test script to verify database functionality
"""
import os
os.environ["USE_DATABASE"] = "true"

from src.database import load_reviews_from_db

def test_database():
    print(" Testing database connection...")

    try:
        # Load a few reviews from database
        reviews = load_reviews_from_db(limit=3)
        print(f"Successfully loaded {len(reviews)} reviews from database")

        # Print first review as sample
        if reviews:
            review = reviews[0]
            print(f"\nSample review:")
            print(f"  ID: {review.review_id}")
            print(f"  Rating: {review.rating}")
            print(f"  Reviewer: {review.reviewer_name}")
            print(f"  Text: {review.review[:100]}...")

        return True

    except Exception as e:
        print(f" Database test failed: {e}")
        return False

if __name__ == "__main__":
    test_database()