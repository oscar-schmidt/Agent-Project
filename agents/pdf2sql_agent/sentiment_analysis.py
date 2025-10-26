"""
Aspect-Based Sentiment Analysis using DeBERTa and PyABSA

This module provides functionality for performing aspect-based sentiment analysis
on text data, extracting specific aspects and their associated sentiments.
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pyabsa import AspectTermExtraction as ATEPC


class SentimentAnalyzer:
    """
    A class for performing aspect-based sentiment analysis on text.

    Attributes:
        confidence_threshold (float): Minimum confidence score for including results
        sentiment_pipeline: Transformers pipeline for sentiment classification
        model_name (str): Name of the pre-trained model to use
    """

    def __init__(self, confidence_threshold=0.8, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        """
        Initialize the sentiment analyzer with specified model and threshold.

        Args:
            confidence_threshold (float): Minimum confidence for results (default: 0.8)
            model_name (str): HuggingFace model name for sentiment analysis
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def basic_sentiment(self, text):
        """
        Perform basic sentiment analysis on a single text.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Sentiment result with label and score
        """
        return self.sentiment_pipeline(text)[0]

    def aspect_sentiment(self, text, aspect):
        """
        Analyze sentiment for a specific aspect within the text.

        Args:
            text (str): The text containing the aspect
            aspect (str): The specific aspect to analyze

        Returns:
            dict: Sentiment result for the specified aspect
        """
        # Format: "text [SEP] aspect" as required by the model
        input_text = f"{text} [SEP] {aspect}"
        return self.sentiment_pipeline(input_text)[0]

    def aspect_sentiment_multiple(self, text, aspects):
        """
        Analyze sentiment for multiple aspects within the same text.

        Args:
            text (str): The text containing the aspects
            aspects (list): List of aspects to analyze

        Returns:
            dict: Dictionary mapping each aspect to its sentiment result
        """
        results = {}
        for aspect in aspects:
            results[aspect] = self.aspect_sentiment(text, aspect)
        return results

    def extract_aspects_with_sentiment(self, texts):
        """
        Automatically extract aspects from texts and analyze their sentiment.
        Uses PyABSA's multilingual aspect extractor.

        Args:
            texts (str or list): Single text or list of texts to analyze

        Returns:
            list: List of dictionaries containing aspect, sentiment, confidence, and sentence
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Initialize aspect extractor
        aspect_extractor = ATEPC.AspectExtractor(
            'multilingual',
            auto_device=True,
            cal_perplexity=True
        )

        # Predict aspects and sentiments
        results = aspect_extractor.predict(
            texts,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True
        )

        # Filter by confidence threshold and format results
        aspect_sentiments = []

        for result in results:
            aspects = result['aspect']
            sentiments = result['sentiment']
            confidences = result['confidence']
            sentence = result['sentence']

            for i in range(len(aspects)):
                if confidences[i] > self.confidence_threshold:
                    aspect_sentiments.append({
                        "aspect": aspects[i],
                        "sentiment": sentiments[i],
                        "confidence": confidences[i],
                        "sentence": sentence
                    })

        return aspect_sentiments


def main():
    """
    Example usage of the SentimentAnalyzer class.

    Demonstrates three different approaches to sentiment analysis:
    1. Basic sentiment analysis on simple text
    2. Aspect-based sentiment for specific features
    3. Automatic aspect extraction from text
    """
    # Initialize analyzer with 80% confidence threshold for filtering results
    analyzer = SentimentAnalyzer(confidence_threshold=0.8)

    # Example 1: Basic sentiment analysis - analyzes overall sentiment of the text
    # without focusing on specific aspects or features
    print("=" * 60)
    print("Example 1: Basic Sentiment Analysis")
    print("=" * 60)
    text1 = "I love this ice-cream"
    result = analyzer.basic_sentiment(text1)
    print(f"Text: {text1}")
    print(f"Sentiment: {result}")
    print()

    # Example 2: Aspect-based sentiment analysis - analyzes sentiment for
    # specific, manually-defined aspects within the same text
    # Useful when you know which aspects you want to analyze
    print("=" * 60)
    print("Example 2: Aspect-Based Sentiment Analysis")
    print("=" * 60)
    text2 = "I love this ice-cream, but the cone is terrible"
    aspects = ["ice-cream", "cone"]  # Manually specify aspects to analyze
    results = analyzer.aspect_sentiment_multiple(text2, aspects)
    print(f"Text: {text2}")
    for aspect, sentiment in results.items():
        print(f"{aspect}: {sentiment}")
    print()

    # Example 3: Automatic aspect extraction with sentiment - uses AI to
    # automatically identify aspects in the text and analyze their sentiment
    # Useful for exploratory analysis when you don't know what aspects exist
    print("=" * 60)
    print("Example 3: Automatic Aspect Extraction")
    print("=" * 60)
    texts = [
        "The food was amazing but the service was slow",
        "Great location and friendly staff, but the room was dirty"
    ]
    # Extract aspects and their sentiments automatically from the texts
    aspects = analyzer.extract_aspects_with_sentiment(texts)
    for item in aspects:
        print(f"Aspect: {item['aspect']}")
        print(f"Sentiment: {item['sentiment']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"Sentence: {item['sentence']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
