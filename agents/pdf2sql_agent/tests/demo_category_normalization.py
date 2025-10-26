"""
Demo script showing category normalization in action
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nodes.category_normalizer import CategoryNormalizer

def demo():
    print("=" * 60)
    print("Category Normalization Demo")
    print("=" * 60)

    normalizer = CategoryNormalizer(
        similarity_threshold=0.75,
        cache_file="demo_categories.json"
    )

    print("\n1. Insurance Reviews - Claim Processing Issues")
    print("-" * 60)

    reviews = [
        ("Review 1", ["Claim Delay"]),
        ("Review 2", ["Lengthy claim process", "Customer Service"]),
        ("Review 3", ["Slow claim handling"]),
        ("Review 4", ["Claim processing time"]),
    ]

    for review_name, categories in reviews:
        normalized = normalizer.normalize_categories(categories)
        print(f"{review_name}: {categories} → {normalized}")

    print("\n2. Restaurant Reviews - Food Quality Issues")
    print("-" * 60)

    restaurant_reviews = [
        ("Review 1", ["Food Quality", "Service"]),
        ("Review 2", ["Food Taste", "Wait Time"]),
        ("Review 3", ["Poor Food Quality"]),
        ("Review 4", ["Service Speed", "Food Temperature"]),
    ]

    for review_name, categories in restaurant_reviews:
        normalized = normalizer.normalize_categories(categories)
        print(f"{review_name}: {categories} → {normalized}")

    print("\n3. Software Reviews - Technical Issues")
    print("-" * 60)

    software_reviews = [
        ("Review 1", ["Authentication Error", "API Performance"]),
        ("Review 2", ["Login Issues", "Slow API"]),
        ("Review 3", ["Auth Problems", "API Response Time"]),
        ("Review 4", ["Session Timeout", "API Latency"]),
    ]

    for review_name, categories in software_reviews:
        normalized = normalizer.normalize_categories(categories)
        print(f"{review_name}: {categories} → {normalized}")

    print("\n4. Category Statistics")
    print("-" * 60)

    all_categories = normalizer.get_all_categories()
    print(f"Total canonical categories: {len(all_categories)}\n")

    for canonical in sorted(all_categories):
        variants = normalizer.get_category_variants(canonical)
        if variants:
            print(f"  {canonical}")
            for variant in variants:
                print(f"    ↳ {variant}")
        else:
            print(f"  {canonical}")

    print("\n" + "=" * 60)
    print("Demo complete! Category mappings saved to demo_categories.json")
    print("=" * 60)

    if os.path.exists("demo_categories.json"):
        print("\nTo reset the demo, delete demo_categories.json")


if __name__ == "__main__":
    demo()
