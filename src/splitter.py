from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    """
    Responsible for splitting large documents into smaller, semantic chunks
    to fit within context windows and improve retrieval accuracy.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into chunks.
        """
        print(f"✂️  Splitting {len(documents)} documents...")
        chunks = self.splitter.split_documents(documents)
        print(f"✅ Generated {len(chunks)} chunks.")
        return chunks