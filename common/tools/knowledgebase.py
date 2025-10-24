from langchain.tools.retriever import create_retriever_tool
from common.utils import IngestKnowledge

retriver_class = IngestKnowledge()
retriver = retriver_class.retrieve()
retriever_tool = create_retriever_tool(
    retriver,
    name="KnowledgeBaseRetriever",
    description="A tool to retrieve information from the knowledge base to answer questions related to the ingested documents."

)