import os
import shutil
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    """
    Manages the Vector Database (ChromaDB) and Embedding Generation (Gemini).
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Initialize ChromaDB (loads from disk if exists, or creates new)
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, chunks: List[Document]):
        """
        Embeds documents and adds them to the vector store.
        """
        if not chunks:
            print("No chunks to add.")
            return
            
        print("Saving embeddings to ChromaDB...")
        self.vector_db.add_documents(chunks)
        print("Data successfully stored in Vector Database.")

    def get_retriever(self, k: int = 4):
        """
        Returns a retriever object for querying the database.
        """
        return self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def clear_database(self):
        """
        Nukes the database directory to start fresh.
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("Vector database cleared.")