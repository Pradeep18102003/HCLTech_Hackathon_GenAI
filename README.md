# HCLTech_Hackathon_GenAI

## 🏗️ System Design

The Mini RAG-powered GenAI Assistant is designed as a modular system with clear separation between:

- **Frontend** (Streamlit Web App)
- **Backend RAG Pipeline** (ingestion, chunking, embedding, retrieval, generation)
- **Vector Store** (ChromaDB)
- **LLM & Embedding Services** (Gemini)

---

### 🔹 High-Level Design

At a high level, the system works in two main phases:

1. **Indexing Phase (Offline / Background)**
   - Users upload documents (PDF, JSON, TXT, URLs, internal docs)
   - Documents are preprocessed, chunked, embedded using Gemini
   - Embeddings + metadata are stored in ChromaDB

2. **Query Phase (Online / User Interaction)**
   - User asks a question via the Streamlit UI
   - Query is embedded using Gemini
   - ChromaDB performs semantic search over stored embeddings
   - Top-k relevant chunks are retrieved as context
   - Context + query are passed to Gemini LLM to generate the final answer

---

### 🔹 Component Design

#### 1. Streamlit Frontend

- Upload multiple documents
- Show upload status and document list
- Provide a text input box for user queries
- Display:
  - Final LLM answer
  - (Optional) Retrieved context snippets
- Trigger backend functions (ingestion / query) via Python callbacks

#### 2. Document Ingestion & Preprocessing Module

**Responsibilities:**
- Accept different file types:
  - `.pdf`, `.json`, `.txt`, URLs, internal dummy docs
- Extract raw text:
  - PDF → text (e.g., using PyPDF)
  - JSON → relevant fields to text
  - TXT → direct reading
  - URL → HTTP GET + HTML parsing (e.g., BeautifulSoup)
- Clean text (remove extra spaces, HTML tags, etc.)
- Split text into **chunks**:
  - Fixed-size windows (e.g., N tokens/characters)
  - Optional overlap for better context retention
- Attach metadata:
  - `source`, `file_name`, `page_number`, `chunk_id`, etc.

#### 3. Embedding & Storage Module

**Responsibilities:**
- Call **Gemini Embeddings API** to convert text chunks into dense vectors
- Create / load a **ChromaDB collection**
- Store:
  - `embedding` (vector)
  - `text` (chunk content)
  - `metadata` (source, page, etc.)

This module abstracts the vector store so that:
- ChromaDB can be swapped out with another DB in the future with minimal changes.

#### 4. Query & Retrieval Module

**Responsibilities:**
- Take user query from the frontend
- Generate **query embedding** using Gemini
- Perform **semantic search** in ChromaDB:
  - `top_k` similar chunks are retrieved based on vector similarity
- Optionally:
  - Rank / filter results further (e.g., based on score threshold)
  - Deduplicate overlapping chunks

The output is a set of **context chunks** that are most relevant to the query.

#### 5. RAG (LLM Orchestration) Module

**Responsibilities:**
- Build a **prompt** for Gemini LLM that contains:
  - User query
  - Retrieved context chunks
  - System instructions (e.g., "Answer only using the given context")
- Send prompt to **Gemini LLM**
- Receive and return the final **answer** text
- Optionally:
  - Return sources / context chunks along with the answer

---

### 🔹 Data Flow

```text
          ┌─────────────────────┐
          │   Streamlit UI      │
          │(upload + ask query) │
          └─────────┬───────────┘
                    │
          ┌─────────▼───────────┐
          │ Ingestion Module     │
          │ (parse, clean,       │
          │  chunk documents)    │
          └─────────┬───────────┘
                    │ chunks
          ┌─────────▼───────────┐
          │ Embedding Module     │
          │  (Gemini Embeddings) │
          └─────────┬───────────┘
                    │ embeddings + metadata
          ┌─────────▼───────────┐
          │   ChromaDB Store     │
          └─────────▲───────────┘
                    │
         User Query │
          ┌─────────┴───────────┐
          │  Query Module        │
          │ (embed + semantic    │
          │  search in ChromaDB) │
          └─────────┬───────────┘
                    │ top-k context
          ┌─────────▼───────────┐
          │   RAG / LLM Module   │
          │ (Gemini LLM answer   │
          │   using context)     │
          └─────────┬───────────┘
                    │
          ┌─────────▼───────────┐
          │   Streamlit UI       │
          │  (show response)     │
          └─────────────────────┘
```
### Folder Structure
mini-rag-assistant/
│
├── app.py                     # Streamlit entry point
├── requirements.txt
├── .env.example               # GEMINI_API_KEY placeholder
│
├── config/
│   └── settings.py            # config (chunk size, top_k, etc.)
│
├── core/
│   ├── ingestion.py           # document loading, cleaning, chunking
│   ├── embeddings.py          # Gemini embedding helpers
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── rag_pipeline.py        # end-to-end RAG orchestration
│   └── utils.py               # shared helpers (logging, text utils)
│
└── data/
    ├── docs/                  # sample documents
    └── chroma_db/             # local ChromaDB persistence (if used)
