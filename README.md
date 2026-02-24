# Smart Contract Document Assistant

A production-ready Python web application for intelligent contract and document analysis using Retrieval-Augmented Generation (RAG). Upload PDF or DOCX contracts and ask natural language questions — the assistant retrieves relevant clauses and generates accurate, grounded answers.

---

## Features

- 📄 **Document Ingestion**: Parse PDF and DOCX files with page-level metadata
- 🔍 **Semantic Search**: Dense vector retrieval using SentenceTransformers (local, no API key needed)
- 🤖 **RAG Pipeline**: Context-grounded Q&A using LangChain and Groq (Llama 3.1 / Mixtral) or HuggingFace LLMs
- 💬 **Multi-turn Conversations**: Session-based conversation history
- 📝 **Document Summarization**: Map-reduce and refine summarization strategies
- 🛡️ **Guardrails**: Safety filtering and automatic legal disclaimers
- 🌐 **REST API**: FastAPI backend with full OpenAPI documentation
- 🖥️ **Web UI**: Gradio interface for upload, chat, and summarization
- 📊 **Evaluation Pipeline**: Built-in retrieval and answer quality metrics with RAGAS integration

---

## Architecture Overview

```
User ──► Gradio UI ──► FastAPI Backend
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Ingestion    Retrieval    Generation
           (Parse,      (Vector      (LLM +
           Chunk,       Search)      RAGChain)
           Embed)
              │             │             │
              └─────────────▼─────────────┘
                       VectorStore
                    (Chroma / FAISS)
```

See [`docs/architecture.md`](docs/architecture.md) for the full component breakdown.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| UI | Gradio |
| LLM Orchestration | LangChain |
| LLM (cloud) | Groq (Llama 3.1, Mixtral, Gemma) |
| LLM (local) | HuggingFace Transformers |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` (local) |
| Vector Store | Chroma / FAISS |
| PDF Parsing | PyMuPDF |
| DOCX Parsing | python-docx |
| Config | pydantic-settings |
| Testing | pytest + unittest.mock |

---

## Prerequisites

- Python 3.9+
- pip
- Groq API key — get a free one at https://console.groq.com/keys
- (Optional) OpenAI API key if using OpenAI as an alternative LLM provider

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd smart_contract_assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
cp .env.example .env
```

---

## Configuration

Edit `.env` to configure the application:

```env
# LLM Configuration
LLM_PROVIDER=groq          # groq | openai | huggingface
MODEL_NAME=llama-3.1-70b-versatile

# Embedding Configuration
# Embeddings run locally — no API key required
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_TYPE=chroma     # chroma | faiss
VECTOR_STORE_PERSIST_DIR=./data/vectorstore

# Groq API Key (required for groq provider)
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY=gsk_...

# OpenAI API Key (optional fallback for openai provider)
# OPENAI_API_KEY=sk-...

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
TOP_K=5
```

---

## Running the Application

### Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Start the Gradio UI

```bash
python ui/gradio_app.py
```

UI available at: http://localhost:7860

---

## API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF/DOCX document |
| POST | `/chat` | Ask a question |
| POST | `/summarize` | Summarize a document |
| DELETE | `/reset` | Clear all data |

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@contract.pdf"
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

### Example: Summarize

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"filename": "contract.pdf"}'
```

---

## Project Structure

```
smart_contract_assistant/
├── app/
│   ├── config.py               # Centralized settings
│   ├── main.py                 # FastAPI application
│   ├── ingestion/
│   │   ├── parser.py           # PDF/DOCX parsing
│   │   ├── chunker.py          # Text chunking
│   │   └── embedder.py         # Embedding generation
│   ├── vectorstore/
│   │   └── store.py            # Chroma/FAISS management
│   ├── retrieval/
│   │   └── retriever.py        # Similarity search
│   ├── generation/
│   │   ├── llm.py              # LLM initialization
│   │   ├── chain.py            # RAG chain
│   │   └── guardrails.py       # Safety & disclaimers
│   ├── conversation/
│   │   └── state.py            # Session state
│   ├── summarization/
│   │   └── summarizer.py       # Document summarization
│   └── evaluation/
│       └── evaluator.py        # Evaluation pipeline
├── ui/
│   └── gradio_app.py           # Gradio web interface
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── docs/
│   ├── architecture.md
│   └── evaluation_report.md
├── data/                       # Uploaded documents & vector store
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_ingestion.py -v

# Run with coverage
pytest --cov=app tests/
```

Tests use `unittest.mock` extensively — no API keys or ML models are required.

---

## Future Enhancements

- [ ] Support for additional file formats (TXT, HTML, Excel)
- [ ] Re-ranking with cross-encoder models
- [ ] Streaming responses for real-time chat
- [ ] Authentication and multi-user support
- [x] RAGAS-based automated evaluation
- [ ] Contract clause extraction and classification
- [ ] Comparison mode for multiple contracts
- [ ] Export conversation history as PDF/DOCX
