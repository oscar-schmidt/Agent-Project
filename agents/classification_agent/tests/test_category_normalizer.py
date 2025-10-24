"""
Tests for category normalization with semantic similarity
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nodes.category_normalizer import CategoryNormalizer


def test_exact_match():
    """Test that exact category names match"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        cat1 = normalizer.normalize_category("Food Quality")
        cat2 = normalizer.normalize_category("Food Quality")

        assert cat1 == cat2 == "Food Quality"

        os.unlink(f.name)


def test_case_insensitive():
    """Test case-insensitive matching"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        cat1 = normalizer.normalize_category("Food Quality")
        cat2 = normalizer.normalize_category("food quality")
        cat3 = normalizer.normalize_category("FOOD QUALITY")

        assert cat1 == cat2 == cat3 == "Food Quality"

        os.unlink(f.name)


def test_semantic_similarity_claims():
    """Test semantic similarity for insurance claim categories"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        cat1 = normalizer.normalize_category("Claim Delay")
        cat2 = normalizer.normalize_category("Lengthy claim process")
        cat3 = normalizer.normalize_category("Slow claim handling")

        # All should map to the first canonical category
        assert cat1 == cat2 == cat3

        variants = normalizer.get_category_variants(cat1)
        assert "Lengthy claim process" in variants
        assert "Slow claim handling" in variants

        os.unlink(f.name)


def test_semantic_similarity_food():
    """Test semantic similarity for restaurant categories"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        cat1 = normalizer.normalize_category("Food Quality")
        cat2 = normalizer.normalize_category("Food Taste")
        cat3 = normalizer.normalize_category("Food Quality Issues")

        # These should match as they're semantically similar
        assert cat1 == cat2 or cat1 == cat3

        os.unlink(f.name)


def test_different_categories():
    """Test that different categories remain separate"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        cat1 = normalizer.normalize_category("Billing Issue")
        cat2 = normalizer.normalize_category("Authentication Error")
        cat3 = normalizer.normalize_category("Performance Problem")

        # These should all be different
        assert cat1 != cat2
        assert cat2 != cat3
        assert cat1 != cat3

        os.unlink(f.name)


def test_normalize_categories_list():
    """Test normalizing a list of categories"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        categories = ["Claim Delay", "Billing Issue", "Lengthy claim process"]
        normalized = normalizer.normalize_categories(categories)

        # Should deduplicate similar categories
        assert len(normalized) == 2
        assert "Billing Issue" in normalized

        os.unlink(f.name)


def test_persistence():
    """Test that category mappings persist across instances"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_file = f.name

        # First instance
        normalizer1 = CategoryNormalizer(cache_file=cache_file, similarity_threshold=0.75)
        cat1 = normalizer1.normalize_category("Claim Delay")
        cat2 = normalizer1.normalize_category("Slow claims")

        assert cat1 == cat2

        # Create new instance with same cache file
        normalizer2 = CategoryNormalizer(cache_file=cache_file, similarity_threshold=0.75)
        cat3 = normalizer2.normalize_category("Lengthy claim process")

        # Should match the persisted canonical category
        assert cat3 == cat1

        os.unlink(cache_file)


def test_threshold_sensitivity():
    """Test that lower threshold is more aggressive in grouping"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        # Low threshold - more aggressive
        normalizer_low = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.6)

        cat1 = normalizer_low.normalize_category("Service")
        cat2 = normalizer_low.normalize_category("Customer Service")

        # With low threshold, these might match
        are_same_low = (cat1 == cat2)

        os.unlink(f.name)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        # High threshold - more strict
        normalizer_high = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.9)

        cat1 = normalizer_high.normalize_category("Service")
        cat2 = normalizer_high.normalize_category("Customer Service")

        # With high threshold, these should be different
        are_same_high = (cat1 == cat2)

        os.unlink(f.name)

    # Low threshold should be more likely to match than high threshold
    assert are_same_low or not are_same_high


def test_merge_categories():
    """Test manual category merging"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        normalizer = CategoryNormalizer(cache_file=f.name, similarity_threshold=0.75)

        # Create two separate categories
        normalizer.normalize_category("Billing")
        normalizer.normalize_category("Invoice")

        # Manually merge them
        success = normalizer.merge_categories("Billing", "Invoice", keep="Billing")

        assert success
        assert normalizer.normalize_category("Invoice") == "Billing"

        os.unlink(f.name)


if __name__ == "__main__":
    print("Running category normalization tests...")

    test_exact_match()
    print("✓ Exact match test passed")

    test_case_insensitive()
    print("✓ Case insensitive test passed")

    test_semantic_similarity_claims()
    print("✓ Semantic similarity (claims) test passed")

    test_semantic_similarity_food()
    print("✓ Semantic similarity (food) test passed")

    test_different_categories()
    print("✓ Different categories test passed")

    test_normalize_categories_list()
    print("✓ List normalization test passed")

    test_persistence()
    print("✓ Persistence test passed")

    test_threshold_sensitivity()
    print("✓ Threshold sensitivity test passed")

    test_merge_categories()
    print("✓ Manual merge test passed")

    print("\n✅ All tests passed!")
