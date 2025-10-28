"""Agent Tools for Review Classification"""

# Import tool classes
from agents.pdf2sql_agent.src.agent.tools.visualization_tool import VisualizationTool
from agents.pdf2sql_agent.src.agent.tools.pdf_tool import PDFTool
from agents.pdf2sql_agent.src.agent.tools.sql_tool import SQLQueryTool

#Import default instances for backward compatibility
from agents.pdf2sql_agent.src.agent.tools.visualization_tool import execute_visualization_tool
from agents.pdf2sql_agent.src.agent.tools.pdf_tool import generate_pdf_report
from agents.pdf2sql_agent.src.agent.tools.sql_tool import execute_sql_query

__all__ = [
    # Tool classes (for direct instantiation with custom config)
    "VisualizationTool",
    "PDFTool",
    "SQLQueryTool",
    # Default instances (for backward compatibility)
    "execute_visualization_tool",
    "generate_pdf_report",
    "execute_sql_query"
]
