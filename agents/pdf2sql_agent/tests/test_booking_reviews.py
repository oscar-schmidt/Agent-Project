"""
Test the category normalization with the three booking reviews from tutor feedback
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import RawReview
from src.nodes.detect_errors import detect_errors_with_ollama
from src.nodes.category_normalizer import get_normalizer

def test_booking_reviews():
    print("=" * 70)
    print("Testing Category Normalization with Booking Reviews")
    print("=" * 70)

    # Reset normalizer for clean test
    normalizer = get_normalizer(similarity_threshold=0.68, reset=True)

    reviews = [
        RawReview(
            review_id="REV-0564",
            review="the time it took to make a booking is ridiculously long. I was so annoyed cause i couldve gotten this done at some place within the half the time.",
            username="max",
            email="max@test.com",
            date="2024-01-01",
            reviewer_name="Max",
            rating=2
        ),
        RawReview(
            review_id="REV-0565",
            review="the process of making a booking was so elongated and it was not efficient at all. I was expecting more from a proffesional company like you.",
            username="mark",
            email="mark@test.com",
            date="2024-01-01",
            reviewer_name="Mark",
            rating=2
        ),
        RawReview(
            review_id="REV-0566",
            review="I had to wait 3 hours to make a booking, that is completely out of this world. 3 hours!!!?? i wish you guys looked in to this because this much traffic is well expected during the holiday season.",
            username="mathew",
            email="mathew@test.com",
            date="2024-01-01",
            reviewer_name="Mathew",
            rating=2
        ),
    ]

    print("\nProcessing reviews...\n")

    all_categories = []
    for review in reviews:
        errors = detect_errors_with_ollama(review)

        print(f"Review {review.review_id} by {review.reviewer_name}")
        print(f"  Text: \"{review.review[:80]}...\"")
        print(f"  Rating: {review.rating}/5")

        for error in errors:
            categories = error.error_type
            all_categories.extend(categories)

            print(f"  Categories: {categories}")
            print(f"  Severity: {error.severity}")
            print(f"  Issue: {error.error_summary}")

        print()

    # Analyze results
    print("=" * 70)
    print("Category Analysis")
    print("=" * 70)

    unique_categories = list(set(all_categories))
    print(f"\nTotal categories used: {len(all_categories)}")
    print(f"Unique canonical categories: {len(unique_categories)}")
    print(f"\nCategories: {unique_categories}")

    # Show category mappings
    print("\n" + "=" * 70)
    print("Category Mappings Learned")
    print("=" * 70)

    for canonical in normalizer.get_all_categories():
        variants = normalizer.get_category_variants(canonical)
        if variants:
            print(f"\n{canonical}")
            for variant in variants:
                print(f"  ↳ {variant}")
        else:
            print(f"\n{canonical} (no variants yet)")

    # Success criteria
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)

    # Count how many times booking/process related categories appear
    booking_related = [c for c in all_categories if any(word in c.lower() for word in ['booking', 'process', 'wait', 'efficiency', 'time'])]

    print(f"\nBooking-related category occurrences: {len(booking_related)}")
    print(f"Unique booking-related categories: {len(set(booking_related))}")

    if len(set(booking_related)) <= 2:
        print("\n✅ SUCCESS: Categories are well-normalized!")
        print("   Similar issues are being grouped together.")
    elif len(set(booking_related)) <= 4:
        print("\n⚠️  PARTIAL: Some normalization happening, but could be better.")
        print(f"   Got {len(set(booking_related))} unique categories for similar issues.")
    else:
        print("\n❌ NEEDS WORK: Too many different categories for similar issues.")
        print(f"   Got {len(set(booking_related))} unique categories when 1-2 expected.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Set environment variable
    os.environ["USE_CATEGORY_NORMALIZATION"] = "true"
    os.environ["CATEGORY_SIMILARITY_THRESHOLD"] = "0.68"

    test_booking_reviews()
