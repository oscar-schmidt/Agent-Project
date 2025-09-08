from langchain_core.tools import BaseTool
import requests
import selenium
from ddgs import DDGS
import json
class WebSearch(BaseTool):
    """Tool class that inherits from base tool"""
    name: str = "WebSearch"
    description: str = "A tool that can search the web"

    def _run(self, query: str) -> str:
        try:
            with DDGS() as search:
                result = list(search.text(query, max_results=3))
            if not result:
                return "no results found"
            return json.dumps(result, indent=2)
        except Exception as e:
            return f'An error occurred during the search: {e}'
