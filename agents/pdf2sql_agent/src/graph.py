import os
import hashlib
from typing import List, Tuple
from langgraph.graph import Graph
from my_tools import execute_sql_query, execute_visualization_tool, generate_pdf_report


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def build_graph() -> Graph:
    g = Graph()

    # def n_load(_: dict) -> List[RawReview]:
    #     use_db = os.getenv("USE_DATABASE", "false").lower() == "true"

    #     if use_db:
    #         #proccessed vs unporcessed stats
    #         stats = get_processing_stats()
    #         print(f" Processing stats: {stats['unprocessed']} unprocessed / {stats['total']} total reviews")

    #         #load unprocessed reviews
    #         data = load_unprocessed_reviews()

    #         if not data:
    #             print("All reviews are up to date")
    #             return []
    #     #DB ERR FALLBACK TO CSV
    #     else:
    #         data = load_reviews(DATA_PATH)
    #         data = data[511:] 
    #         print(f"Loaded {len(data)} reviews from CSV")

    #     return data

    # #detect errors with LLM
    # def n_detect(reviews: List[RawReview]) -> List[Tuple[RawReview, List[DetectedError]]]:
    #     pairs: List[Tuple[RawReview, List[DetectedError]]] = []
    #     for r in reviews:
    #         errs = detect_errors_with_ollama(r, OLLAMA_MODEL)
    #         pairs.append((r, errs))
    #     return pairs

    # #analyze sentiment for each review
    # def n_sentiment(pairs: List[Tuple[RawReview, List[DetectedError]]]) -> List[Tuple[RawReview, List[DetectedError], SentimentData]]:
    #     """
    #     Enrich error pairs with sentiment analysis.

    #     INPUT: List[Tuple[RawReview, List[DetectedError]]]
    #     OUTPUT: List[Tuple[RawReview, List[DetectedError], SentimentData]]
    #     """
    #     # Check if sentiment is enabled
    #     if os.getenv("ENABLE_SENTIMENT", "true").lower() == "false":
    #         # Return with dummy sentiment data
    #         print("Sentiment analysis disabled (ENABLE_SENTIMENT=false)")
    #         return [(r, errs, SentimentData(
    #             review_id=r.review_id,
    #             overall_sentiment="Neutral",
    #             overall_confidence=0.0,
    #             sentiment_polarity=0.0
    #         )) for r, errs in pairs]

    #     print(f"Analyzing sentiment for {len(pairs)} reviews...")
    #     enriched = []

    #     for idx, (review, errors) in enumerate(pairs, 1):
    #         sentiment = analyze_review_sentiment(review)
    #         enriched.append((review, errors, sentiment))

    #         if idx % 10 == 0:
    #             print(f"  ... {idx}/{len(pairs)} analyzed")

    #     print("Sentiment analysis complete")
    #     return enriched

    # #normalise and classify based on severity (now with sentiment)
    # def n_normalize(enriched_pairs: List[Tuple[RawReview, List[DetectedError], SentimentData]]) -> List[EnrichedError]:
    #     out: List[EnrichedError] = []
    #     for review, errors, sentiment in enriched_pairs:
    #         out.extend(normalize(review, errors, sentiment))
    #     return out

    # #write to Notion and mark as processed
    # def n_tee(items: List[EnrichedError]) -> List[EnrichedError]:
    #     if not items:
    #         return items

    #     dry = os.getenv("NOTION_DRY_RUN", "0") in ("1", "true", "True")
    #     use_db = os.getenv("USE_DATABASE", "false").lower() == "true"
    #     processed_review_ids = []

    #     for i, e in enumerate(items, 1):
    #         hash_value = getattr(e, "error_hash", None) or _sha12(
    #             f"{e.review.review_id}|{e.error.error_summary}"
    #         )
    #         if dry:
    #             print(f"[dry-run] Would upsert {e.review.review_id} | {e.error.error_summary} | {hash_value}")
    #         else:
    #             upsert_enriched_error(e)

    #         #review IDs for batch processing
    #         if use_db:
    #             processed_review_ids.append(e.review.review_id)

    #         if i % 20 == 0:
    #             print(f"… processed {i} rows")

    #     #mark all reviews as processed in batch
    #     if use_db and processed_review_ids:
    #         mark_reviews_processed(processed_review_ids)

       
    #     return items

    import json

    def n_query(_: dict):
        result = json.loads(execute_sql_query._run(question="Show me total sales by department"))
        return result

    def n_visualize(query_result: dict):
        df = query_result["dataframe"]
        vis_result = json.loads(execute_visualization_tool._run(df=df, query=query_result["query"], question="Show me total sales by department"
            )
        )
        return vis_result

    def n_report(vis_result: dict):
        df = vis_result["dataframe"]
        report_result = json.loads(generate_pdf_report._run(df=df, query="Show me total sales by department", question="Show me total sales by department", graph=True))
        return report_result




    # # Nodes
    # g.add_node("load", n_load)
    # g.add_node("detect", n_detect)
    # g.add_node("sentiment", n_sentiment)  # Sentiment analysis node
    # g.add_node("normalize", n_normalize)
    # g.add_node("tee", n_tee)

    # # Edges - Sequential flow with sentiment between detect and normalize
    # g.add_edge("load", "detect")
    # g.add_edge("detect", "sentiment")      # detect -> sentiment
    # g.add_edge("sentiment", "normalize")   # sentiment -> normalize
    # g.add_edge("normalize", "tee")

    g.add_node("query", n_query)
    g.add_node("visualize", n_visualize)
    g.add_node("report", n_report)

    g.add_edge("query", "visualize")
    g.add_edge("visualize", "report")

    # Entry & finish
    g.set_entry_point("query")
    g.set_finish_point("report")

    return g


# export workflow for run.py
wf = build_graph().compile()
