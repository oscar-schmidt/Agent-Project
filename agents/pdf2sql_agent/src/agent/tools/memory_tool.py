from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional

from agents.pdf2sql_agent.src.memory.qdrant_store import QdrantStore


class MemorySearchInput(BaseModel):
    """input for memory search"""
    query: str = Field(description="what to search for (eg 'crash keywords', 'payment issues')")
    top_k: int = Field(default=5, description="how many memories to get")


def search_memory(query: str, top_k: int = 5) -> str:
    """search long term memory for past experiences and knowledge"""
    try:
        store = QdrantStore(collection_name="ReviewAgent")
        memories = store.get(query=query, top_k=top_k)

        if not memories:
            return "no relevant memories found"

        formatted = []
        for i, mem in enumerate(memories, 1):
            mem_type = mem["memory_type"]
            data = mem["memory_data"]
            score = mem["score"]

            if mem_type == "Episode":
                formatted.append(
                    f"{i}. Episode (relevance: {score:.2f})\n"
                    f"   what: {data['observation']}\n"
                    f"   why: {data['thoughts']}\n"
                    f"   result: {data['result']}"
                )
            elif mem_type == "Semantic":
                ctx = f" ({data['context']})" if data.get('context') else ""
                formatted.append(
                    f"{i}. Fact (relevance: {score:.2f})\n"
                    f"   {data['subject']} -> {data['predicate']} -> {data['object']}{ctx}"
                )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"memory search failed: {str(e)}\nnote: make sure qdrant is running"


# create the tool
memory_search_tool = StructuredTool.from_function(
    func=search_memory,
    name="MemorySearch",
    description=(
        "search long term memory for past classification cases, learned patterns, "
        "user preferences, and rules. use when recalling similar cases or checking "
        "if youve seen similar keywords before"
    ),
    args_schema=MemorySearchInput
)
