from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
import pandas as pd

class IngestKnowledge():
    """A class to ingest knowledge from PDF files into a Qdrant vector store."""

    def __init__(self):
        """Initialize the Qdrant vector store."""
        client = QdrantClient()
        embedings = OpenAIEmbeddings()
        embedding_size = len(embedings.embed_query("test"))
        try:
            client.get_collection("knowledgebase")
        except Exception as e:
            client.create_collection(
                collection_name="knowledgebase",
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE
                )
            )

        self.vector_store = QdrantVectorStore(
            client=QdrantClient(url="http://localhost:6333"),
            collection_name="knowledgebase",
            embedding=OpenAIEmbeddings(),
        )
    async def ingest_exel(self, exel_path: str):
        cols = ['ReviewID', 'Reviewer', 'Date', 'ReviewText',
                'ErrorSummary', 'ErrorType', 'Criticality', 'Rationale']

        df = pd.read_excel(exel_path, usecols=cols)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pd.DataFrame(df))
        self.vector_store.add_documents(docs)
        print(f"Ingested {len(docs)} documents from {exel_path} into Exel.")

    async def ingest_pdf(self, pdf_path: str):
        """Ingest a PDF file into the vector store."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        docs = text_splitter.split_documents(documents)

        self.vector_store.add_documents(docs)
        print(f"Ingested {len(docs)} documents from {pdf_path} into Qdrant.")

    def retrieve(self):
        """Retrieve documents from the vector store."""

        return self.vector_store.as_retriever(k=5)