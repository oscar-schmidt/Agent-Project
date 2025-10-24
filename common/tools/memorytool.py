from langchain_core.tools import BaseTool
from typing_extensions import override
from common.stores.QdrantStore import QdrantStore
class MemoryTool(BaseTool):
    name: str = "MemorySearch"
    description: str = "A tool that can search through your long term memory for semantic or episodic memories. Input should be a search query."
    @override
    def  _run(self, query: str):
        store = QdrantStore(collection_name="WebAgent")
        memories = store.get(query)
        return memories