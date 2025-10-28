from agents.pdf2sql_agent.src.graph import wf

if __name__ == "__main__":
    enriched = wf.invoke({})  
    print(f"\n Done. Processed {len(enriched)} enriched errors.\n")
    print("────────────────────────────────────────────────────────────")

    for idx, e in enumerate(enriched, 1):
        # extra safety in case something upstream changes
        if not hasattr(e, "review"):
            continue
        print(f"Review #{idx} ({e.review.review_id}, rating={e.review.rating})")
        print(f"Text: {e.review.review}")

        # NEW: Display sentiment information
        if hasattr(e, 'sentiment_data') and e.sentiment_data:
            print(f" → Sentiment: {e.sentiment_data.overall_sentiment} (confidence: {e.sentiment_data.overall_confidence:.2%}, polarity: {e.sentiment_data.sentiment_polarity:+.3f})")

        print(f" → Severity: {e.criticality}")

        # NEW: Show if sentiment influenced criticality
        if hasattr(e, 'sentiment_influenced_criticality') and e.sentiment_influenced_criticality:
            print(f" → ⚠️  Criticality adjusted by SQL2PDFAgent")

        print(f" → Categories: {e.error.error_type}")
        print(f" → Summary: {e.error.error_summary}")
        print(f" → Rationale: {e.error.rationale}")
        print()
