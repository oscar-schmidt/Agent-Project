"""
Category Normalization Service
Ensures semantic consistency across review categories using embedding-based similarity.
"""

import os
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class CategoryNormalizer:
    """
    Normalizes categories using semantic similarity.
    Maps similar categories (e.g., "Claim Delay", "Lengthy claim process") to a canonical name.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        cache_file: str = None
    ):
        self.model = SentenceTransformer(model)
        self.similarity_threshold = similarity_threshold

        if cache_file is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "category_mappings.json")

        self.cache_file = cache_file
        self.categories: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.categories = json.load(f)
                    print(f"[CategoryNormalizer] Loaded {len(self.categories)} category mappings")
            except Exception as e:
                print(f"[CategoryNormalizer] Failed to load cache: {e}")

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.categories, f, indent=2)
        except Exception as e:
            print(f"[CategoryNormalizer] Failed to save cache: {e}")

    def _compute_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _find_similar_category(self, category: str, embedding: List[float]) -> Optional[str]:
        best_match = None
        best_similarity = 0.0

        for canonical_name, data in self.categories.items():
            canonical_embedding = data["embedding"]
            similarity = self._cosine_similarity(embedding, canonical_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = canonical_name

        if best_similarity >= self.similarity_threshold:
            return best_match

        return None

    def normalize_category(self, category: str) -> str:
        category = category.strip()
        if not category:
            return "Other"

        # Check exact match first
        category_lower = category.lower()
        for canonical_name, data in self.categories.items():
            if canonical_name.lower() == category_lower:
                return canonical_name

            if category_lower in [v.lower() for v in data.get("variants", [])]:
                return canonical_name

        # Semantic matching
        embedding = self._compute_embedding(category)
        similar_category = self._find_similar_category(category, embedding)

        if similar_category:
            if category not in self.categories[similar_category].get("variants", []):
                self.categories[similar_category].setdefault("variants", []).append(category)
                self._save_cache()
            return similar_category

        # Create new canonical category
        self.categories[category] = {
            "embedding": embedding,
            "variants": []
        }
        self._save_cache()

        return category

    def normalize_categories(self, categories: List[str]) -> List[str]:
        normalized = []
        seen = set()

        for category in categories:
            canonical = self.normalize_category(category)
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)

        return normalized or ["Other"]

    def get_all_categories(self) -> List[str]:
        return sorted(self.categories.keys())

    def get_category_variants(self, canonical_name: str) -> List[str]:
        return self.categories.get(canonical_name, {}).get("variants", [])

    def merge_categories(self, category1: str, category2: str, keep: str = None) -> bool:
        if category1 not in self.categories or category2 not in self.categories:
            return False

        if keep is None:
            keep = category1

        discard = category2 if keep == category1 else category1

        keep_data = self.categories[keep]
        discard_data = self.categories[discard]

        keep_data.setdefault("variants", []).append(discard)

        for variant in discard_data.get("variants", []):
            if variant not in keep_data["variants"]:
                keep_data["variants"].append(variant)

        del self.categories[discard]

        self._save_cache()
        return True

    def reset(self):
        self.categories = {}
        self._save_cache()

    def set_similarity_threshold(self, threshold: float):
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")


_normalizer_instance = None


def get_normalizer(similarity_threshold: float = None, reset: bool = False) -> CategoryNormalizer:
    global _normalizer_instance

    if _normalizer_instance is None:
        threshold = similarity_threshold or float(os.getenv("CATEGORY_SIMILARITY_THRESHOLD", "0.75"))
        _normalizer_instance = CategoryNormalizer(similarity_threshold=threshold)
    elif similarity_threshold is not None:
        _normalizer_instance.set_similarity_threshold(similarity_threshold)

    if reset:
        _normalizer_instance.reset()

    return _normalizer_instance
