# Architecture Documentation

## Overview

The Smart Contract Assistant is a Retrieval-Augmented Generation (RAG) system designed to help users analyze and query legal contracts and documents. Users can upload PDF or DOCX files, which are processed and indexed for semantic search. Questions are answered using retrieved document context and a large language model.

---

## Component Descriptions

### 1. Ingestion Pipeline (`app/ingestion/`)

Responsible for processing uploaded documents into searchable chunks.

- **Parser** (`parser.py`): Reads raw PDF (via PyMuPDF) and DOCX (via python-docx) files, extracting text with page-level metadata.
- **Chunker** (`chunker.py`): Splits extracted text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` to respect token limits.
- **Embedder** (`embedder.py`): Generates dense vector embeddings for each chunk using either SentenceTransformers (local) or OpenAI's embedding API.

### 2. Vector Store (`app/vectorstore/`)

Stores and retrieves document embeddings.

- **VectorStoreManager** (`store.py`): Abstracts over Chroma (persistent, local) and FAISS (in-memory) vector databases. Provides `add_documents`, `similarity_search`, `clear`, and `get_retriever` methods.

### 3. Retrieval (`app/retrieval/`)

Finds relevant document chunks for a given query.

- **DocumentRetriever** (`retriever.py`): Wraps `VectorStoreManager` to retrieve top-k similar chunks and format them as a context string for the LLM.

### 4. Generation (`app/generation/`)

Produces answers using retrieved context and an LLM.

- **LLM Interface** (`llm.py`): Constructs a LangChain-compatible LLM (ChatOpenAI or HuggingFacePipeline) from configuration.
- **RAGChain** (`chain.py`): Combines the retriever and LLM with a structured prompt template to perform grounded question answering.
- **Guardrails** (`guardrails.py`): Filters harmful queries, appends legal disclaimers, and annotates factuality.

### 5. Conversation (`app/conversation/`)

Manages multi-turn dialogue state.

- **ConversationState** (`state.py`): Stores per-session message history and formats it for inclusion in prompts.

### 6. Summarization (`app/summarization/`)

Produces document-level summaries.

- **DocumentSummarizer** (`summarizer.py`): Uses LangChain's `load_summarize_chain` with map-reduce or refine strategies.

### 7. API (`app/main.py`)

FastAPI application exposing REST endpoints.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload and index a document |
| `/chat` | POST | Ask a question about documents |
| `/summarize` | POST | Summarize an uploaded document |
| `/reset` | DELETE | Clear vector store and sessions |

### 8. UI (`ui/gradio_app.py`)

Gradio-based web interface with two tabs:
- **Upload Document**: File upload, status display, and summarization.
- **Chat**: Multi-turn conversational Q&A.

### 9. Evaluation (`app/evaluation/`)

Measures system performance.

- **RAGEvaluator** (`evaluator.py`): Evaluates retrieval (average docs retrieved) and generation (answer length, result logging) quality.

---

## System Flow Diagram

```
User
 │
 ├─[Upload File]──► Parser ──► Chunker ──► Embedder ──► VectorStore
 │                                                           │
 └─[Ask Question]──► Guardrails ──► DocumentRetriever ───────┘
                                         │
                                         ▼
                              Context + Prompt Template
                                         │
                                         ▼
                                     LLM (RAGChain)
                                         │
                                         ▼
                             Answer + Disclaimer ──► User
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI |
| Web Server | Uvicorn |
| UI | Gradio |
| LLM Orchestration | LangChain |
| LLM (cloud) | OpenAI GPT-3.5 / GPT-4 |
| LLM (local) | HuggingFace Transformers |
| Embeddings (local) | SentenceTransformers |
| Embeddings (cloud) | OpenAI Embeddings |
| Vector Store | Chroma / FAISS |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Configuration | pydantic-settings |
| Testing | pytest + unittest.mock |
