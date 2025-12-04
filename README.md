# HCLTech_Hackathon_GenAI
A lightweight Retrieval-Augmented Generation (RAG) based AI assistant capable of accepting **multiple document formats** (PDFs, JSON, text files, articles, public PDFs, or internal dummy documents).  
The system preprocesses, chunks, embeds, stores, and retrieves relevant information to answer user queries with high accuracy using **Gemini + ChromaDB + Streamlit**.

---

## Overview

This project implements a **mini RAG (Retrieval-Augmented Generation)** architecture to build an intelligent assistant that can reason over multiple documents.  
Users can upload different types of files, ask questions, and the assistant will fetch relevant document context using vector search and generate responses using **Gemini LLM**.

---

## Key Features

- Accepts **multiple document formats**:
  - PDFs  
  - JSON  
  - TXT files  
  - Public URLs / Articles  
  - Internal dummy documents  
- Automatic **preprocessing and chunking**  
- Semantic embeddings generated using **Gemini embeddings API**  
- Vector storage and retrieval with **ChromaDB**  
- **Semantic search** over uploaded documents  
- Context injected into Gemini LLM for accurate responses  
- Clean, interactive **Streamlit web interface**  
- Fully deployable via **Streamlit Cloud**

---

## Architecture
User Query
↓
Gemini Text Embedding
↓
Vector Search (ChromaDB)
↓ Top-k Context
Document Store ↑
↓
RAG Pipeline → Gemini LLM → Final Response


---

## Workflow / Approach

### 1. Document Ingestion
Users can upload multiple types of documents.  
We handle:
- PDF extraction  
- JSON parsing  
- Text file loading  
- URL scraping (for public articles)

---

### 2. Preprocessing & Chunking
All documents undergo:
- Cleaning  
- Splitting into smaller chunks (configurable size)  
- Metadata assignment (source, page number, etc.)

---

### 3. Embedding Generation
Each chunk is embedded using:

 **Gemini Embeddings API**

These embeddings create numerical representations usable for semantic search.

---

### 4. Storage in Vector Database

We use **ChromaDB** to store embeddings + metadata.

---

### 5. Semantic Search

When a user enters a query:

- Query → Embedded using Gemini  
- ChromaDB performs **top-k nearest neighbor search**  
- Retrieve the most relevant chunks as **context**

---

### 6. RAG Response Generation

Context + user query → passed into **Gemini LLM**,  
which generates a factual, grounded response based only on the retrieved documents.

---

### 7. Deployment

Frontend is built using **Streamlit**.  
The app can be deployed easily using **Streamlit Cloud**.

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| LLM | Gemini |
| Embeddings | Gemini Embeddings API |
| Vector DB | ChromaDB |
| Frontend UI | Streamlit |
| Backend | Python |
| Document Handling | PyPDF, JSON, Requests, BeautifulSoup |

---

## How to Run

### 1. Clone the repository
```bash
git clone <repo-link>
cd mini-rag-assistant
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Add your API Key
Create a .env file
```bash
GEMINI_API_KEY=your_key_here
```
### 4. Run the Streamlit app
```bash
streamlit run app.py
```

