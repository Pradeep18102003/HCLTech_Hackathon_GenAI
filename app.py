import streamlit as st
import os
import tempfile
import shutil

# Import our backend modules
from src.loader import DataLoader
from src.splitter import TextChunker
from src.embedding import VectorStoreManager
from src.rag import RAGApplication
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="Gemini RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Load Env
load_dotenv()

# --- HEADER ---
st.title("🤖 Chat with Your Data (Gemini RAG)")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("1. Configuration")
    
    # API Key Handling
    api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    
    st.header("2. Add Data Sources")
    
    # Option A: File Upload
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, JSON)", 
        type=["pdf", "txt", "json"], 
        accept_multiple_files=True
    )
    
    # Option B: Web URL
    web_url = st.text_input("Enter Website URL")
    
    # Process Button
    if st.button("Process & Ingest Data", type="primary"):
        if not api_key:
            st.error("Please enter a Google API Key first.")
        elif not uploaded_files and not web_url:
            st.error("Please provide a file or a URL.")
        else:
            with st.spinner("Processing... This may take a moment."):
                try:
                    # 1. Initialize Managers
                    vector_manager = VectorStoreManager()
                    loader = DataLoader()
                    chunker = TextChunker()
                    
                    documents = []
                    
                    # 2. Handle Files
                    if uploaded_files:
                        # Create a temp directory to save uploaded files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            for uploaded_file in uploaded_files:
                                temp_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Load using our existing loader
                                docs = loader.load_source(temp_path)
                                documents.extend(docs)
                    
                    # 3. Handle URL
                    if web_url:
                        docs = loader.load_source(web_url)
                        documents.extend(docs)
                    
                    # 4. Split and Embed
                    if documents:
                        chunks = chunker.split_documents(documents)
                        vector_manager.add_documents(chunks)
                        st.success(f"Successfully ingested {len(documents)} documents ({len(chunks)} chunks)!")
                    else:
                        st.warning("No text could be extracted from the provided sources.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.divider()
    
    # Reset Button
    if st.button("🗑️ Clear Database"):
        try:
            vector_manager = VectorStoreManager()
            vector_manager.clear_database()
            st.success("Database cleared!")
            st.session_state.messages = [] # Clear chat history
        except Exception as e:
            st.error(f"Error clearing database: {e}")

# --- MAIN CHAT AREA ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload some documents or a URL in the sidebar, and then ask me anything about them."}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            if not api_key:
                response = "❌ Error: Google API Key is missing. Please add it in the sidebar."
            else:
                # Initialize RAG App
                vector_manager = VectorStoreManager()
                rag = RAGApplication(vector_manager)
                
                # Get Answer
                response = rag.query(prompt)
        except Exception as e:
            response = f"❌ Error: {str(e)}"
        
        message_placeholder.markdown(response)
    
    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response})