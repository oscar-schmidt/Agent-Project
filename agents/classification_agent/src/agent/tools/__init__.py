"""Agent Tools for Review Classification"""

# Import tool classes
from agents.classification_agent.src.agent.tools.criticality_tool import CriticalityTool
from agents.classification_agent.src.agent.tools.sentiment_tool import SentimentTool
from agents.classification_agent.src.agent.tools.notion_tool import NotionTool
from agents.classification_agent.src.agent.tools.date_time import DateTime

#Import default instances for backward compatibility
from agents.classification_agent.src.agent.tools.criticality_tool import classify_review_criticality
from agents.classification_agent.src.agent.tools.sentiment_tool import analyze_review_sentiment
from agents.classification_agent.src.agent.tools.notion_tool import log_reviews_to_notion
from agents.classification_agent.src.agent.tools.date_time import get_current_datetime

__all__ = [
    # Tool classes (for direct instantiation with custom config)
    "CriticalityTool",
    "SentimentTool",
    "NotionTool",
    "DateTime",
    # Default instances (for backward compatibility)
    "classify_review_criticality",
    "analyze_review_sentiment",
    "log_reviews_to_notion",
    "get_current_datetime"
]
