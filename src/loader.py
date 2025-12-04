import os
import json
from typing import List, Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
    WebBaseLoader
)
from langchain_core.documents import Document

class DataLoader:
    """
    A unified interface for loading documents from various sources.
    """

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Loads a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Loads a plain text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"📄 Loading Text file: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()

    @staticmethod
    def load_json(file_path: str, jq_schema: str = ".") -> List[Document]:
        """
        Loads a JSON file. 
        Args:
            file_path: Path to the JSON file.
            jq_schema: The jq schema to extract text fields (default: all).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"📄 Loading JSON: {file_path}")
        loader = JSONLoader(file_path, jq_schema=jq_schema, text_content=False)
        return loader.load()

    @staticmethod
    def load_website(url: str) -> List[Document]:
        """Scrapes a website."""
        print(f"🌐 Loading Website: {url}")
        loader = WebBaseLoader(url)
        return loader.load()

    def load_source(self, source: str) -> List[Document]:
        """
        Detects source type and loads data accordingly.
        """
        if source.startswith("http"):
            return self.load_website(source)
        elif source.endswith(".pdf"):
            return self.load_pdf(source)
        elif source.endswith(".txt"):
            return self.load_text(source)
        elif source.endswith(".json"):
            return self.load_json(source)
        else:
            raise ValueError(f"Unsupported file type: {source}")