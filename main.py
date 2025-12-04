import argparse
import os
from dotenv import load_dotenv

# Import our custom modules
from src.loader import DataLoader
from src.splitter import TextChunker
from src.embedding import VectorStoreManager
from src.rag import RAGApplication

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Gemini RAG Application")
    
    # Mode selection
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into the database")
    parser.add_argument("--query", type=str, help="Ask a question to the RAG system")
    
    # Ingestion arguments
    parser.add_argument("--source", action="append", help="Path to PDF, TXT, JSON or URL (can be used multiple times)")
    parser.add_argument("--clear", action="store_true", help="Clear existing database before ingesting")

    args = parser.parse_args()

    # Verify API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return

    # Initialize Core Components
    vector_manager = VectorStoreManager()
    
    # --- INGESTION PIPELINE ---
    if args.ingest:
        if args.clear:
            vector_manager.clear_database()
            # Re-initialize after clearing
            vector_manager = VectorStoreManager()

        if not args.source:
            print("Please provide at least one source using --source")
            return

        loader = DataLoader()
        chunker = TextChunker()

        all_documents = []
        for src in args.source:
            try:
                raw_docs = loader.load_source(src)
                all_documents.extend(raw_docs)
            except Exception as e:
                print(f"Failed to load {src}: {e}")

        if all_documents:
            chunks = chunker.split_documents(all_documents)
            vector_manager.add_documents(chunks)
        else:
            print("No documents were loaded successfully.")

    # --- QUERY PIPELINE ---
    elif args.query:
        rag_app = RAGApplication(vector_manager)
        answer = rag_app.query(args.query)
        
        print("\n" + "="*50)
        print("Answer:")
        print(answer)
        print("="*50 + "\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()