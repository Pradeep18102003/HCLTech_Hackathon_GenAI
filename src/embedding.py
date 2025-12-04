import os
import shutil
import gc
import time
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
        
        # Initialize ChromaDB
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, chunks: List[Document]):
        """
        Embeds documents and adds them to the vector store.
        """
        if not chunks:
            print("⚠️ No chunks to add.")
            return
            
        print("💾 Saving embeddings to ChromaDB...")
        self.vector_db.add_documents(chunks)
        print("✅ Data successfully stored in Vector Database.")

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
        Safely clears the database by forcing garbage collection first.
        """
        print("🗑️  Attempting to clear database...")
        
        # 1. Delete the Chroma object from memory to release the file handle
        if hasattr(self, 'vector_db'):
            # This is critical for Windows: explicitly remove the object
            del self.vector_db
            self.vector_db = None
        
        # 2. Force Python Garbage Collection to verify files are released
        gc.collect()
        
        # 3. Add a tiny delay to let the OS file system catch up
        time.sleep(1.0)

        # 4. Delete the folder
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print("✅ Vector database cleared.")
            except PermissionError:
                print("❌ PermissionError: Windows is still holding the file lock.")
                print("👉 TIP: Stop the terminal (Ctrl+C) and delete the 'chroma_db' folder manually.")
            except Exception as e:
                print(f"❌ Error deleting database: {e}")